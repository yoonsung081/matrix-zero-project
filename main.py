print("\n[System] Loading Evo-Matrix AI Engine v2.1 (Pure RL Mode)...")
import os
import sys
import gc

if 'VIRTUAL_ENV' in os.environ and 'CONDA_PREFIX' in os.environ:
    os.environ.pop('CONDA_PREFIX', None)

if sys.platform != 'win32':
    if 'CUDA_PATH' not in os.environ:
        for path in ['/usr/local/cuda', '/usr/local/cuda-13', '/usr/local/cuda-12', '/usr/local/cuda-11']:
            if os.path.exists(path):
                os.environ['CUDA_PATH'] = path
                break
else:
    if 'CUDA_PATH' not in os.environ:
        base_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
        if os.path.exists(base_path):
            versions = sorted([d for d in os.listdir(base_path) if d.startswith('v')], reverse=True)
            if versions:
                cuda_dir = os.path.join(base_path, versions[0])
                os.environ['CUDA_PATH'] = cuda_dir
                os.environ['PATH'] = os.path.join(cuda_dir, 'bin') + os.pathsep + \
                                     os.path.join(cuda_dir, 'libnvvp') + os.pathsep + \
                                     os.environ['PATH']

try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, AttributeError, RuntimeError):
    HAS_GPU = False

if not HAS_GPU:
    try:
        import numpy as cp
        print("Warning: GPU (CuPy) unavailable. Running on CPU mode (NumPy).")
    except ImportError:
        print("\n[Critical Error] NumPy is not installed. Please run: pip install numpy")
        sys.exit(1)

def to_cpu(array):
    if HAS_GPU and hasattr(array, 'get'):
        return array.get()
    return array

import numpy as np
import json
import random
from tqdm import tqdm

PRIV_NAMES = ["행/열 교환", "원소 0으로 만들기", "두 원소 위치 교환"]

# ── 특징 벡터 차원 ──────────────────────────────────────────────
# [0:9]   내 행렬 (9)
# [9:12]  x 벡터 (3)
# [12]    라운드 (1)
# [13:19] 전체 팀 점수 정규화 (6)
# [19:25] 전체 팀 행렬식 정규화 (6)
# [25]    내 순위 정규화 0=1위 1=꼴찌 (1)
# [26]    편향 bias (1)
# [27:30] 패딩 (3)
FEAT_DIM = 30
ACT_DIM  = 18   # 9칸 × ±1
PRIV_DIM = 18   # 3종류 × 6대상


class MatrixGame:
    def __init__(self):
        self.num_players = 6
        self.matrices   = [cp.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector   = cp.zeros((3, 1), dtype=int)
        self.current_round = 1

        # 가중치 (Pure RL — 휴리스틱 없음)
        self.weights      = cp.random.randn(FEAT_DIM, ACT_DIM)  * 0.01
        self.priv_weights = cp.random.randn(FEAT_DIM, PRIV_DIM) * 0.01

        self.learning_rate = 0.001
        # 분산 감소용 Baseline (지수이동평균)
        self.reward_baseline = 0.0
        self.baseline_alpha  = 0.005

        # self-play 이전 세대 저장
        self.learner_history = []

    # ── 기본 게임 로직 ────────────────────────────────────────────

    def calculate_x(self):
        stacked = cp.stack(self.matrices)
        diags   = cp.diagonal(stacked, axis1=1, axis2=2)
        self.x_vector = cp.floor(cp.sum(diags, axis=0) / self.num_players).astype(int).reshape(3, 1)

    def get_score(self, player_idx, mode="auto"):
        r   = self.current_round
        m_f = self.matrices[player_idx].astype(cp.float32)
        if mode == "ax" or (mode == "auto" and r <= 2):
            return float(to_cpu(cp.sum(cp.dot(m_f, self.x_vector.astype(cp.float32)))))
        else:
            return float(to_cpu(cp.round(cp.linalg.det(m_f))))

    def get_round_winner(self):
        scores    = [self.get_score(i) for i in range(self.num_players)]
        score_map = {}
        for idx, s in enumerate(scores):
            score_map.setdefault(s, []).append(idx)
        for s in sorted(score_map.keys(), reverse=True)[:3]:
            if len(score_map[s]) == 1:
                return score_map[s][0]
        return None

    def get_final_winner(self):
        survivors = []
        for i in range(self.num_players):
            det = self.get_score(i, mode="det")
            if det != 0:
                survivors.append((self.get_score(i, mode="ax"), abs(det), i))
        if not survivors:
            return None
        return sorted(survivors, key=lambda x: (x[0], x[1]), reverse=True)[0][2]

    def get_reward(self, player_idx):
        if self.current_round == 5:
            det    = self.get_score(player_idx, mode="det")
            ax_sum = self.get_score(player_idx, mode="ax")
            if det == 0:
                return -1000.0
            is_winner = (self.get_final_winner() == player_idx)
            return (ax_sum * 10) + (5000 if is_winner else 0)
        return 100.0 if self.get_round_winner() == player_idx else 0.0

    def reset_game(self):
        self.matrices      = [cp.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector      = cp.zeros((3, 1), dtype=int)
        self.current_round = 1

    # ── 특징 벡터 ─────────────────────────────────────────────────

    def _get_feature(self, p_idx):
        """
        30차원 순수 게임 상태 인코딩.
        휴리스틱 계산 없이 관측 가능한 값만 사용.
        """
        feat = np.zeros(FEAT_DIM, dtype=np.float32)

        # 내 행렬
        feat[0:9] = to_cpu(self.matrices[p_idx].flatten()).astype(np.float32)

        # x 벡터
        feat[9:12] = to_cpu(self.x_vector.flatten()).astype(np.float32)

        # 라운드
        feat[12] = float(self.current_round)

        # 전체 팀 점수 (정규화)
        scores = np.array([self.get_score(i) for i in range(self.num_players)], dtype=np.float32)
        denom  = np.max(np.abs(scores)) + 1e-8
        feat[13:19] = scores / denom

        # 전체 팀 행렬식 (정규화)
        dets = np.array([
            float(to_cpu(cp.round(cp.linalg.det(self.matrices[i].astype(cp.float32)))))
            for i in range(self.num_players)
        ], dtype=np.float32)
        denom_det = np.max(np.abs(dets)) + 1e-8
        feat[19:25] = dets / denom_det

        # 내 순위 (0=1위, 1=꼴찌)
        my_score = float(scores[p_idx])
        rank = sum(1 for s in scores if s > my_score) + 1
        feat[25] = (rank - 1) / max(self.num_players - 1, 1)

        # bias
        feat[26] = 1.0

        return cp.array(feat)

    # ── softmax 탐험 ──────────────────────────────────────────────

    def _softmax_sample(self, logits, temperature):
        """
        온도 기반 softmax 샘플링.
        temperature → 0 : greedy,  temperature → ∞ : random
        """
        l = logits / max(temperature, 1e-6)
        l = l - cp.max(l)                  # 수치 안정성
        probs = cp.exp(l)
        probs = probs / cp.sum(probs)
        probs_cpu = to_cpu(probs).astype(np.float64)
        probs_cpu = probs_cpu / probs_cpu.sum()   # 부동소수점 오차 보정
        return int(np.random.choice(len(probs_cpu), p=probs_cpu))

    # ── 에이전트 ──────────────────────────────────────────────────

    def agent_random(self, p_idx):
        """완전 랜덤 에이전트 (학습 상대용)"""
        r, c = random.randint(0, 2), random.randint(0, 2)
        self.matrices[p_idx][r, c] += random.choice([-1, 1])

    def agent_learner_act(self, p_idx, weights, temperature=0.0):
        """
        학습 에이전트 행동.
        temperature > 0 : softmax 탐험
        temperature = 0 : greedy (평가/대전 시)
        """
        feat = self._get_feature(p_idx)
        logits = cp.dot(feat, weights)

        if temperature > 0:
            action_idx = self._softmax_sample(logits, temperature)
        else:
            action_idx = int(to_cpu(cp.argmax(logits)))

        r_idx = (action_idx // 2) // 3
        c_idx = (action_idx // 2) % 3
        v     = 1 if action_idx % 2 == 0 else -1
        self.matrices[p_idx][r_idx, c_idx] += v
        return feat, action_idx

    def agent_learner_privilege(self, winner_idx, temperature=0.0):
        """
        학습 에이전트 특권 선택.
        특권 액션: priv_type(0~2) * 6 + target_idx(0~5)
        """
        feat   = self._get_feature(winner_idx)
        logits = cp.dot(feat, self.priv_weights)

        if temperature > 0:
            priv_action_idx = self._softmax_sample(logits, temperature)
        else:
            priv_action_idx = int(to_cpu(cp.argmax(logits)))

        priv_type  = priv_action_idx // 6
        target_idx = priv_action_idx % 6
        minimize   = (target_idx != winner_idx)

        effect = self._best_priv_position(priv_type, target_idx, minimize)
        self._apply_priv(priv_type, target_idx, effect)
        return feat, priv_action_idx

    def agent_random_privilege(self, winner_idx):
        """랜덤 에이전트의 특권: 유형·대상 랜덤, 위치 greedy"""
        priv_type  = random.randint(0, 2)
        target_idx = random.randint(0, self.num_players - 1)
        minimize   = (target_idx != winner_idx)
        effect = self._best_priv_position(priv_type, target_idx, minimize)
        self._apply_priv(priv_type, target_idx, effect)

    # ── 특권 위치 탐색 / 적용 ────────────────────────────────────

    def _eval_matrix(self, m_matrix):
        m_f = m_matrix.astype(cp.float32)
        if self.current_round <= 2:
            return float(to_cpu(cp.sum(cp.dot(m_f, self.x_vector.astype(cp.float32)))))
        return float(to_cpu(cp.round(cp.linalg.det(m_f))))

    def _best_priv_position(self, priv_type, target_idx, minimize=False):
        m = self.matrices[target_idx]
        best_effect = None
        best_score  = float('inf') if minimize else -float('inf')

        def is_better(s):
            return s < best_score if minimize else s > best_score

        if priv_type == 0:
            for i, j in [(0,1),(0,2),(1,2)]:
                for axis in ('row','col'):
                    m_new = m.copy()
                    if axis == 'row':
                        tmp = m_new[i].copy(); m_new[i] = m_new[j]; m_new[j] = tmp
                    else:
                        tmp = m_new[:,i].copy(); m_new[:,i] = m_new[:,j]; m_new[:,j] = tmp
                    s = self._eval_matrix(m_new)
                    if is_better(s):
                        best_score = s; best_effect = (axis, i, j)

        elif priv_type == 1:
            for r in range(3):
                for c in range(3):
                    if int(to_cpu(m[r, c])) != 0:
                        m_new = m.copy(); m_new[r, c] = 0
                        s = self._eval_matrix(m_new)
                        if is_better(s):
                            best_score = s; best_effect = (r, c)

        elif priv_type == 2:
            pos = [(r, c) for r in range(3) for c in range(3)]
            for k in range(len(pos)):
                for l in range(k+1, len(pos)):
                    r1,c1 = pos[k]; r2,c2 = pos[l]
                    if int(to_cpu(m[r1,c1])) == int(to_cpu(m[r2,c2])):
                        continue
                    m_new = m.copy()
                    tmp = m_new[r1,c1].copy()
                    m_new[r1,c1] = m_new[r2,c2]; m_new[r2,c2] = tmp
                    s = self._eval_matrix(m_new)
                    if is_better(s):
                        best_score = s; best_effect = (r1,c1,r2,c2)

        return best_effect

    def _apply_priv(self, priv_type, target_idx, effect):
        if effect is None:
            return
        m = self.matrices[target_idx]
        if priv_type == 0:
            axis, i, j = effect
            if axis == 'row':
                tmp = m[i].copy(); m[i] = m[j]; m[j] = tmp
            else:
                tmp = m[:,i].copy(); m[:,i] = m[:,j]; m[:,j] = tmp
        elif priv_type == 1:
            r, c = effect; m[r, c] = 0
        elif priv_type == 2:
            r1,c1,r2,c2 = effect
            tmp = m[r1,c1].copy(); m[r1,c1] = m[r2,c2]; m[r2,c2] = tmp

    # ── 가중치 업데이트 ───────────────────────────────────────────

    def update_weights(self, episode_actions, episode_privs, final_reward):
        """
        REINFORCE + Baseline (advantage = reward - EMA baseline).
        휴리스틱 없이 순수 정책 그래디언트.
        """
        # Baseline 업데이트 (지수이동평균)
        self.reward_baseline += self.baseline_alpha * (final_reward - self.reward_baseline)
        advantage = final_reward - self.reward_baseline

        gamma = 0.95

        # 일반 행동
        for feat, action_idx, r_num in episode_actions:
            discount = gamma ** (5 - r_num)
            grad = cp.outer(feat, cp.eye(ACT_DIM, dtype=cp.float32)[action_idx])
            self.weights += self.learning_rate * advantage * discount * grad
        self.weights = cp.clip(self.weights, -1, 1)

        # 특권 행동
        for feat, priv_action_idx, r_num in episode_privs:
            discount = gamma ** (5 - r_num)
            grad = cp.outer(feat, cp.eye(PRIV_DIM, dtype=cp.float32)[priv_action_idx])
            self.priv_weights += self.learning_rate * advantage * discount * grad
        self.priv_weights = cp.clip(self.priv_weights, -1, 1)

    # ── 시뮬레이션 ────────────────────────────────────────────────

    def run_simulation(self, num_episodes=30000):
        """
        Pure RL Self-play.
        상대: 완전 랜덤(V1) + 이전 세대 자신 — 휴리스틱 에이전트 없음.
        탐험: softmax temperature 1.0 → 0.05 선형 감소.
        """
        print(f"Starting Pure RL Simulation: {num_episodes} episodes")
        print("상대 구성: Random + Self-play  |  탐험: softmax temperature decay\n")
        results_data = []

        # 초기 히스토리
        self.learner_history.append((self.weights.copy(), self.priv_weights.copy()))

        TEMP_START = 1.0
        TEMP_END   = 0.05

        for ep in tqdm(range(num_episodes)):
            self.reset_game()
            episode_actions = []
            episode_privs   = []

            # 온도 선형 감소 (탐험 → 활용)
            temperature = TEMP_START - (TEMP_START - TEMP_END) * (ep / num_episodes)

            # 1000판마다 현재 가중치 스냅샷 저장
            if ep % 1000 == 0:
                self.learner_history.append((self.weights.copy(), self.priv_weights.copy()))

            # 상대 구성: 랜덤 또는 이전 세대 — 휴리스틱 없음
            opponents = []
            for _ in range(5):
                opponents.append("Random" if random.random() < 0.3 else "OldSelf")

            for r in range(1, 6):
                self.current_round = r

                for p in range(self.num_players):
                    for _ in range(r):
                        if p == 0:  # 학습 에이전트
                            feat, a_idx = self.agent_learner_act(p, self.weights, temperature)
                            episode_actions.append((feat, a_idx, r))
                        else:
                            opp = opponents[p - 1]
                            if opp == "Random":
                                self.agent_random(p)
                            else:
                                old_w, _ = random.choice(self.learner_history)
                                # 이전 세대는 greedy (평가 모드)
                                self.agent_learner_act(p, old_w, temperature=0.0)

                self.calculate_x()

                # 특권 처리 (3종류 모두 반영)
                winner = self.get_round_winner()
                if winner is not None:
                    if winner == 0:
                        feat, priv_a = self.agent_learner_privilege(winner, temperature)
                        episode_privs.append((feat, priv_a, r))
                    else:
                        self.agent_random_privilege(winner)

            final_reward = self.get_reward(0)
            self.update_weights(episode_actions, episode_privs, final_reward)

            if HAS_GPU and ep % 500 == 0:
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

            if ep > num_episodes - 1000:
                results_data.append({
                    "matrix": to_cpu(self.matrices[0]).tolist(),
                    "x":      to_cpu(self.x_vector).flatten().tolist(),
                    "score":  self.get_score(0)
                })

        with open("strategy_weights.json", "w") as f:
            json.dump({
                "weights":      to_cpu(self.weights).tolist(),
                "priv_weights": to_cpu(self.priv_weights).tolist(),
                "feat_dim":     FEAT_DIM,
                "sample_data":  results_data
            }, f)
        print("\nTraining Complete. Weights saved to strategy_weights.json")

    # ── 저장 / 로드 ───────────────────────────────────────────────

    def load_weights(self):
        filename = "strategy_weights.json"
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            saved_dim = data.get("feat_dim", 20)
            if saved_dim != FEAT_DIM:
                print(f"  [경고] 저장된 특징 차원({saved_dim}) ≠ 현재({FEAT_DIM}). 재학습 필요.")
                return False

            self.weights      = cp.array(data["weights"])
            if "priv_weights" in data:
                self.priv_weights = cp.array(data["priv_weights"])
            print("  가중치 로드 완료 (일반 + 특권)")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

    # ── 대화형 대전 ───────────────────────────────────────────────

    def _human_choose_privilege(self, winner_idx):
        print("\n[PRIVILEGE] 사용할 특권:")
        for i, name in enumerate(PRIV_NAMES):
            print(f"  {i+1}. {name}")
        while True:
            try:
                priv_type = int(input("특권 종류 (1/2/3): ")) - 1
                if priv_type in (0, 1, 2):
                    break
            except ValueError:
                pass
        while True:
            try:
                target_idx = int(input("대상 팀 번호 (1~6): ")) - 1
                if 0 <= target_idx < self.num_players:
                    break
            except ValueError:
                pass

        minimize = (target_idx != winner_idx)
        effect   = self._best_priv_position(priv_type, target_idx, minimize)
        if effect is None:
            print("  적용 가능한 효과 없음.")
            return
        self._apply_priv(priv_type, target_idx, effect)
        direction = "약화" if minimize else "강화"
        print(f"  → Team {target_idx+1}에 '{PRIV_NAMES[priv_type]}' ({direction})")
        print(f"  결과 행렬:\n{to_cpu(self.matrices[target_idx])}")

    def play_interactive(self):
        self.reset_game()
        print("\n" + "="*50)
        print(" [MATCH] AI (Team 1) vs YOU (Team 2) vs Random Bots")
        print("="*50)

        for r in range(1, 6):
            self.current_round = r
            print(f"\n{'='*20} ROUND {r} {'='*20}")

            for p in range(self.num_players):
                if p == 0:  # AI — greedy (평가 모드)
                    for _ in range(r):
                        self.agent_learner_act(p, self.weights, temperature=0.0)
                    print("[AI] 행동 완료.")
                elif p == 1:  # Human
                    print(f"\n[YOUR TURN] 현재 행렬:\n{to_cpu(self.matrices[p])}")
                    for a in range(r):
                        while True:
                            try:
                                move = input(f"  행동 {a+1}/{r} (행 열 값, 예: 1 1 1): ").split()
                                row, col, val = int(move[0]), int(move[1]), int(move[2])
                                if val not in [-1, 1] or not (0 <= row <= 2 and 0 <= col <= 2):
                                    raise ValueError
                                self.matrices[p][row, col] += val
                                break
                            except Exception:
                                print("  형식 오류. 행(0-2) 열(0-2) 값(-1 또는 1)")
                else:  # Random bots
                    for _ in range(r):
                        self.agent_random(p)

            self.calculate_x()
            scores = [self.get_score(i) for i in range(self.num_players)]
            print(f"\n공통 X 벡터: {to_cpu(self.x_vector).flatten()}")
            print("현재 점수:", {f"T{i+1}": round(s, 1) for i, s in enumerate(scores)})

            winner = self.get_round_winner()
            if winner is not None:
                labels = {0: "AI(T1)", 1: "YOU(T2)"}
                print(f"\n라운드 승자: {labels.get(winner, f'Bot(T{winner+1})')}")
                if winner == 0:
                    _, priv_a = self.agent_learner_privilege(winner, temperature=0.0)
                    pt, ti = priv_a // 6, priv_a % 6
                    print(f"[AI 특권] Team {ti+1}에 '{PRIV_NAMES[pt]}' ({'약화' if ti != winner else '강화'})")
                    print(f"  결과 행렬:\n{to_cpu(self.matrices[ti])}")
                elif winner == 1:
                    self._human_choose_privilege(winner)
                else:
                    self.agent_random_privilege(winner)
                    print("[Bot 특권] 랜덤 특권 적용")

        print("\n" + "="*50)
        print(" [최종 결과]")
        winner_idx = self.get_final_winner()
        for i in range(self.num_players):
            label  = "AI(T1)" if i == 0 else "YOU(T2)" if i == 1 else f"Bot(T{i+1})"
            det    = self.get_score(i, mode="det")
            ax_sum = self.get_score(i, mode="ax")
            status = "WINNER" if i == winner_idx else "SURVIVED" if det != 0 else "ELIMINATED"
            print(f" {label}: Ax={ax_sum:.1f}  det={det}  [{status}]")
        print("="*50)
        if winner_idx is None:
            print(" 결과: 전원 탈락")
        else:
            labels = {0: "AI (Team 1)", 1: "YOU (Team 2)"}
            print(f" 최종 승자: {labels.get(winner_idx, f'Bot (Team {winner_idx+1})')}")


if __name__ == "__main__":
    game = MatrixGame()
    print("=== Evo-Matrix AI Engine (Pure RL) ===")
    print("1. 대전 테스트 (AI vs 인간)")
    print("2. 학습 (30,000 에피소드)")

    choice = input("\n선택 (1 또는 2): ").strip()
    if choice == "1":
        if game.load_weights():
            game.play_interactive()
        else:
            print("\n[오류] 학습된 가중치가 없습니다. 먼저 옵션 2를 실행하세요.")
    elif choice == "2":
        game.run_simulation(30000)
    else:
        print("잘못된 선택.")
