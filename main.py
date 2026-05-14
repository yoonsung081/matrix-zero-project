print("\n[System] Loading Evo-Matrix AI Engine v3.0 (MLP + Action Masking + League)...")
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

PRIV_NAMES = ["행 교환", "열 교환", "원소 0으로 만들기", "두 원소 위치 교환"]

FEAT_DIM   = 30
HIDDEN_DIM = 64   # 개선1: MLP 은닉층
ACT_DIM    = 18
PRIV_DIM   = 8


class MatrixGame:
    def __init__(self):
        self.num_players = 6
        self.matrices    = [cp.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector    = cp.zeros((3, 1), dtype=int)
        self.current_round = 1

        # 개선1: 선형 → MLP (일반 행동)
        self.w1 = cp.random.randn(FEAT_DIM,   HIDDEN_DIM) * 0.01
        self.b1 = cp.zeros(HIDDEN_DIM)
        self.w2 = cp.random.randn(HIDDEN_DIM, ACT_DIM)    * 0.01
        self.b2 = cp.zeros(ACT_DIM)

        # 개선1: 선형 → MLP (특권 행동)
        self.pw1 = cp.random.randn(FEAT_DIM,   HIDDEN_DIM) * 0.01
        self.pb1 = cp.zeros(HIDDEN_DIM)
        self.pw2 = cp.random.randn(HIDDEN_DIM, PRIV_DIM)   * 0.01
        self.pb2 = cp.zeros(PRIV_DIM)

        self.learning_rate   = 0.001
        self.reward_baseline = 0.0
        self.baseline_alpha  = 0.005

        # 개선2: 라운드별 사용 칸 추적 (액션 마스킹)
        self.used_cells = [set() for _ in range(self.num_players)]

        # 개선3: 리그 히스토리 스냅샷
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
                survivors.append((self.get_score(i, mode="ax"), det, i))
        if not survivors:
            return []
        best_ax = max(s[0] for s in survivors)
        top_ax  = [s for s in survivors if s[0] == best_ax]
        if len(top_ax) == 1:
            return [top_ax[0][2]]
        best_det = max(s[1] for s in top_ax)
        return [s[2] for s in top_ax if s[1] == best_det]

    def get_reward(self, player_idx):
        if self.current_round == 5:
            det = self.get_score(player_idx, mode="det")
            if det == 0:
                return -1000.0
            return 5000.0 if player_idx in self.get_final_winner() else 0.0
        return 100.0 if self.get_round_winner() == player_idx else 0.0

    def reset_game(self):
        self.matrices      = [cp.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector      = cp.zeros((3, 1), dtype=int)
        self.current_round = 1
        for s in self.used_cells:
            s.clear()

    # ── 특징 벡터 ─────────────────────────────────────────────────

    def _get_feature(self, p_idx):
        feat = np.zeros(FEAT_DIM, dtype=np.float32)
        feat[0:9]  = to_cpu(self.matrices[p_idx].flatten()).astype(np.float32)
        feat[9:12] = to_cpu(self.x_vector.flatten()).astype(np.float32)
        feat[12]   = float(self.current_round)

        scores    = np.array([self.get_score(i) for i in range(self.num_players)], dtype=np.float32)
        denom     = np.max(np.abs(scores)) + 1e-8
        feat[13:19] = scores / denom

        dets = np.array([
            float(to_cpu(cp.round(cp.linalg.det(self.matrices[i].astype(cp.float32)))))
            for i in range(self.num_players)
        ], dtype=np.float32)
        denom_det   = np.max(np.abs(dets)) + 1e-8
        feat[19:25] = dets / denom_det

        my_score = float(scores[p_idx])
        rank = sum(1 for s in scores if s > my_score) + 1
        feat[25] = (rank - 1) / max(self.num_players - 1, 1)
        feat[26]  = 1.0

        return cp.array(feat)

    # ── 개선1: MLP 순전파 ─────────────────────────────────────────

    def _mlp_forward(self, feat, w1, b1, w2, b2):
        h = cp.maximum(0, cp.dot(feat, w1) + b1)   # ReLU
        return cp.dot(h, w2) + b2, h

    def _current_mlp(self):
        return (self.w1, self.b1, self.w2, self.b2)

    # ── 개선2: 액션 마스킹 ────────────────────────────────────────

    def _get_action_mask(self, p_idx):
        mask = cp.ones(ACT_DIM, dtype=cp.float32)
        for r, c in self.used_cells[p_idx]:
            cell_idx = r * 3 + c
            mask[cell_idx * 2]     = 0.0
            mask[cell_idx * 2 + 1] = 0.0
        return mask

    # ── softmax 탐험 ──────────────────────────────────────────────

    def _softmax_sample(self, logits, temperature):
        l = logits / max(temperature, 1e-6)
        l = l - cp.max(l)
        probs = cp.exp(l)
        probs = probs / cp.sum(probs)
        probs_cpu = to_cpu(probs).astype(np.float64)
        probs_cpu = probs_cpu / probs_cpu.sum()
        return int(np.random.choice(len(probs_cpu), p=probs_cpu))

    # ── 에이전트 ──────────────────────────────────────────────────

    def agent_random(self, p_idx):
        # 개선2: 이미 사용한 칸 제외
        avail = [(r, c) for r in range(3) for c in range(3)
                 if (r, c) not in self.used_cells[p_idx]]
        if not avail:
            avail = [(r, c) for r in range(3) for c in range(3)]
        r, c = random.choice(avail)
        self.matrices[p_idx][r, c] += random.choice([-1, 1])
        self.used_cells[p_idx].add((r, c))

    def agent_learner_act(self, p_idx, mlp, temperature=0.0):
        w1, b1, w2, b2 = mlp
        feat = self._get_feature(p_idx)

        # 개선1: MLP 순전파
        logits, h = self._mlp_forward(feat, w1, b1, w2, b2)

        # 개선2: 사용한 칸 마스킹
        mask   = self._get_action_mask(p_idx)
        logits = logits + (1.0 - mask) * cp.float32(-1e9)

        if temperature > 0:
            action_idx = self._softmax_sample(logits, temperature)
        else:
            action_idx = int(to_cpu(cp.argmax(logits)))

        r_idx = (action_idx // 2) // 3
        c_idx = (action_idx // 2) % 3
        v     = 1 if action_idx % 2 == 0 else -1
        self.matrices[p_idx][r_idx, c_idx] += v
        self.used_cells[p_idx].add((r_idx, c_idx))
        return feat, h, action_idx

    def _best_priv_position_global(self, priv_type, winner_idx):
        others      = [t for t in range(self.num_players) if t != winner_idx]
        best_effect = None
        best_total  = float('inf')

        if priv_type == 0:
            for i, j in [(0,1),(0,2),(1,2)]:
                total = 0
                for t in others:
                    m_new = self.matrices[t].copy()
                    tmp = m_new[i].copy(); m_new[i] = m_new[j]; m_new[j] = tmp
                    total += self._eval_matrix(m_new)
                if total < best_total:
                    best_total = total; best_effect = (i, j)

        elif priv_type == 1:
            for i, j in [(0,1),(0,2),(1,2)]:
                total = 0
                for t in others:
                    m_new = self.matrices[t].copy()
                    tmp = m_new[:,i].copy(); m_new[:,i] = m_new[:,j]; m_new[:,j] = tmp
                    total += self._eval_matrix(m_new)
                if total < best_total:
                    best_total = total; best_effect = (i, j)

        elif priv_type == 2:
            for r in range(3):
                for c in range(3):
                    total = 0
                    for t in others:
                        m_new = self.matrices[t].copy()
                        m_new[r, c] = 0
                        total += self._eval_matrix(m_new)
                    if total < best_total:
                        best_total = total; best_effect = (r, c)

        elif priv_type == 3:
            pos = [(r, c) for r in range(3) for c in range(3)]
            for k in range(len(pos)):
                for l in range(k+1, len(pos)):
                    r1,c1 = pos[k]; r2,c2 = pos[l]
                    total = 0
                    for t in others:
                        m_new = self.matrices[t].copy()
                        tmp = m_new[r1,c1].copy()
                        m_new[r1,c1] = m_new[r2,c2]; m_new[r2,c2] = tmp
                        total += self._eval_matrix(m_new)
                    if total < best_total:
                        best_total = total; best_effect = (r1,c1,r2,c2)

        return best_effect

    def apply_privilege(self, winner_idx, priv_type, scope):
        if scope == 0:
            effect = self._best_priv_position(priv_type, winner_idx, minimize=False)
            self._apply_priv(priv_type, winner_idx, effect)
        else:
            effect = self._best_priv_position_global(priv_type, winner_idx)
            for t in range(self.num_players):
                if t != winner_idx:
                    self._apply_priv(priv_type, t, effect)

    def agent_learner_privilege(self, winner_idx, temperature=0.0):
        feat = self._get_feature(winner_idx)
        logits, h = self._mlp_forward(feat, self.pw1, self.pb1, self.pw2, self.pb2)

        if temperature > 0:
            priv_action_idx = self._softmax_sample(logits, temperature)
        else:
            priv_action_idx = int(to_cpu(cp.argmax(logits)))

        priv_type = priv_action_idx // 2
        scope     = priv_action_idx % 2

        self.apply_privilege(winner_idx, priv_type, scope)
        return feat, h, priv_action_idx

    def agent_random_privilege(self, winner_idx):
        priv_type = random.randint(0, 2)
        scope     = random.randint(0, 1)
        self.apply_privilege(winner_idx, priv_type, scope)

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
                m_new = m.copy()
                tmp = m_new[i].copy(); m_new[i] = m_new[j]; m_new[j] = tmp
                s = self._eval_matrix(m_new)
                if is_better(s): best_score = s; best_effect = (i, j)

        elif priv_type == 1:
            for i, j in [(0,1),(0,2),(1,2)]:
                m_new = m.copy()
                tmp = m_new[:,i].copy(); m_new[:,i] = m_new[:,j]; m_new[:,j] = tmp
                s = self._eval_matrix(m_new)
                if is_better(s): best_score = s; best_effect = (i, j)

        elif priv_type == 2:
            for r in range(3):
                for c in range(3):
                    if int(to_cpu(m[r, c])) != 0:
                        m_new = m.copy(); m_new[r, c] = 0
                        s = self._eval_matrix(m_new)
                        if is_better(s): best_score = s; best_effect = (r, c)

        elif priv_type == 3:
            pos = [(r, c) for r in range(3) for c in range(3)]
            for k in range(len(pos)):
                for l in range(k+1, len(pos)):
                    r1,c1 = pos[k]; r2,c2 = pos[l]
                    if int(to_cpu(m[r1,c1])) == int(to_cpu(m[r2,c2])): continue
                    m_new = m.copy()
                    tmp = m_new[r1,c1].copy()
                    m_new[r1,c1] = m_new[r2,c2]; m_new[r2,c2] = tmp
                    s = self._eval_matrix(m_new)
                    if is_better(s): best_score = s; best_effect = (r1,c1,r2,c2)

        return best_effect

    def _apply_priv(self, priv_type, target_idx, effect):
        if effect is None: return
        m = self.matrices[target_idx]
        if priv_type == 0:
            i, j = effect; tmp = m[i].copy(); m[i] = m[j]; m[j] = tmp
        elif priv_type == 1:
            i, j = effect; tmp = m[:,i].copy(); m[:,i] = m[:,j]; m[:,j] = tmp
        elif priv_type == 2:
            r, c = effect; m[r, c] = 0
        elif priv_type == 3:
            r1,c1,r2,c2 = effect
            tmp = m[r1,c1].copy(); m[r1,c1] = m[r2,c2]; m[r2,c2] = tmp

    # ── 개선1: MLP REINFORCE 역전파 ───────────────────────────────

    def update_weights(self, episode_actions, episode_privs, final_reward):
        self.reward_baseline += self.baseline_alpha * (final_reward - self.reward_baseline)
        advantage = final_reward - self.reward_baseline
        gamma     = 0.95

        for feat, h, action_idx, r_num in episode_actions:
            scale = self.learning_rate * advantage * (gamma ** (5 - r_num))
            e_a   = cp.zeros(ACT_DIM, dtype=cp.float32)
            e_a[action_idx] = 1.0

            d_h = cp.dot(self.w2, e_a) * (h > 0)   # w2 업데이트 전에 계산

            self.w2 += scale * cp.outer(h, e_a)
            self.b2 += scale * e_a
            self.w1 += scale * cp.outer(feat, d_h)
            self.b1 += scale * d_h

        for arr in [self.w1, self.b1, self.w2, self.b2]:
            cp.clip(arr, -1, 1, out=arr)

        for feat, h, priv_action_idx, r_num in episode_privs:
            scale = self.learning_rate * advantage * (gamma ** (5 - r_num))
            e_a   = cp.zeros(PRIV_DIM, dtype=cp.float32)
            e_a[priv_action_idx] = 1.0

            d_h = cp.dot(self.pw2, e_a) * (h > 0)

            self.pw2 += scale * cp.outer(h, e_a)
            self.pb2 += scale * e_a
            self.pw1 += scale * cp.outer(feat, d_h)
            self.pb1 += scale * d_h

        for arr in [self.pw1, self.pb1, self.pw2, self.pb2]:
            cp.clip(arr, -1, 1, out=arr)

    # ── 개선3: 리그 스냅샷 ────────────────────────────────────────

    def _snapshot(self):
        return {
            'w1': self.w1.copy(), 'b1': self.b1.copy(),
            'w2': self.w2.copy(), 'b2': self.b2.copy(),
        }

    # ── 시뮬레이션 ────────────────────────────────────────────────

    def run_simulation(self, num_episodes=30000):
        print(f"Starting MLP RL Simulation: {num_episodes} episodes")
        print("개선: MLP 정책 / 액션 마스킹 / 리그 상대 (Random 20% / 이전세대 30% / 최신자신 50%)\n")
        results_data = []

        self.learner_history.append(self._snapshot())

        TEMP_START = 1.0
        TEMP_END   = 0.05

        for ep in tqdm(range(num_episodes)):
            self.reset_game()
            episode_actions = []
            episode_privs   = []

            temperature = TEMP_START - (TEMP_START - TEMP_END) * (ep / num_episodes)

            if ep % 1000 == 0:
                self.learner_history.append(self._snapshot())

            # 개선3: 리그 상대 구성
            opponents = []
            for _ in range(5):
                rv = random.random()
                if rv < 0.2:
                    opponents.append("Random")
                elif rv < 0.5 and len(self.learner_history) > 1:
                    opponents.append("League")   # 랜덤 이전 세대 챔피언
                else:
                    opponents.append("OldSelf")  # 가장 최근 스냅샷

            for r in range(1, 6):
                self.current_round = r
                for s in self.used_cells:   # 라운드 시작 시 사용 칸 초기화
                    s.clear()

                for p in range(self.num_players):
                    for _ in range(r):
                        if p == 0:
                            feat, h, a_idx = self.agent_learner_act(p, self._current_mlp(), temperature)
                            episode_actions.append((feat, h, a_idx, r))
                        else:
                            opp = opponents[p - 1]
                            if opp == "Random":
                                self.agent_random(p)
                            elif opp == "League":
                                snap = random.choice(self.learner_history)
                                self.agent_learner_act(p, (snap['w1'], snap['b1'], snap['w2'], snap['b2']))
                            else:
                                snap = self.learner_history[-1]
                                self.agent_learner_act(p, (snap['w1'], snap['b1'], snap['w2'], snap['b2']))

                self.calculate_x()

                winner = self.get_round_winner()
                if winner is not None:
                    if winner == 0:
                        feat, h, priv_a = self.agent_learner_privilege(winner, temperature)
                        episode_privs.append((feat, h, priv_a, r))
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
                "w1": to_cpu(self.w1).tolist(),  "b1": to_cpu(self.b1).tolist(),
                "w2": to_cpu(self.w2).tolist(),  "b2": to_cpu(self.b2).tolist(),
                "pw1": to_cpu(self.pw1).tolist(), "pb1": to_cpu(self.pb1).tolist(),
                "pw2": to_cpu(self.pw2).tolist(), "pb2": to_cpu(self.pb2).tolist(),
                "feat_dim":   FEAT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "sample_data": results_data
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

            if data.get("feat_dim", 0) != FEAT_DIM or data.get("hidden_dim", 0) != HIDDEN_DIM:
                print(f"  [경고] 저장된 차원이 현재와 다릅니다 (feat={data.get('feat_dim')}, hidden={data.get('hidden_dim')}). 재학습 필요.")
                return False
            if "w1" not in data:
                print("  [경고] 구버전 가중치 형식 (선형 모델). 재학습 필요.")
                return False

            self.w1  = cp.array(data["w1"]);  self.b1  = cp.array(data["b1"])
            self.w2  = cp.array(data["w2"]);  self.b2  = cp.array(data["b2"])
            self.pw1 = cp.array(data["pw1"]); self.pb1 = cp.array(data["pb1"])
            self.pw2 = cp.array(data["pw2"]); self.pb2 = cp.array(data["pb2"])
            print("  MLP 가중치 로드 완료 (일반 + 특권)")
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
                if priv_type in (0, 1, 2): break
            except ValueError:
                pass

        print("  적용 범위:")
        print("  1. 나에게만 (내 행렬 강화)")
        print("  2. 전체 타팀 (모든 상대 행렬 약화)")
        while True:
            try:
                scope = int(input("범위 선택 (1/2): ")) - 1
                if scope in (0, 1): break
            except ValueError:
                pass

        self.apply_privilege(winner_idx, priv_type, scope)
        if scope == 0:
            print(f"  → 내 행렬에 '{PRIV_NAMES[priv_type]}' 적용 (강화)")
            print(f"  결과:\n{to_cpu(self.matrices[winner_idx])}")
        else:
            print(f"  → 전체 타팀에 '{PRIV_NAMES[priv_type]}' 적용 (약화)")
            for t in range(self.num_players):
                if t != winner_idx:
                    print(f"  Team {t+1}:\n{to_cpu(self.matrices[t])}")

    def play_interactive(self):
        self.reset_game()
        print("\n" + "="*50)
        print(" [MATCH] AI (Team 1) vs YOU (Team 2) vs Random Bots")
        print("="*50)

        for r in range(1, 6):
            self.current_round = r
            for s in self.used_cells:
                s.clear()
            print(f"\n{'='*20} ROUND {r} {'='*20}")

            for p in range(self.num_players):
                if p == 0:
                    for _ in range(r):
                        self.agent_learner_act(p, self._current_mlp(), temperature=0.0)
                    print("[AI] 행동 완료.")
                elif p == 1:
                    print(f"\n[YOUR TURN] 현재 행렬:\n{to_cpu(self.matrices[p])}")
                    for a in range(r):
                        while True:
                            try:
                                move = input(f"  행동 {a+1}/{r} (행 열 값, 예: 1 1 1): ").split()
                                row, col, val = int(move[0]), int(move[1]), int(move[2])
                                if val not in [-1, 1] or not (0 <= row <= 2 and 0 <= col <= 2):
                                    raise ValueError
                                if (row, col) in self.used_cells[p]:
                                    print("  이미 이번 라운드에 사용한 칸입니다. 다른 칸을 선택하세요.")
                                    continue
                                self.matrices[p][row, col] += val
                                self.used_cells[p].add((row, col))
                                break
                            except Exception:
                                print("  형식 오류. 행(0-2) 열(0-2) 값(-1 또는 1)")
                else:
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
                    _, _, priv_a = self.agent_learner_privilege(winner, temperature=0.0)
                    pt    = priv_a // 2
                    scope = priv_a % 2
                    if scope == 0:
                        print(f"[AI 특권] 나에게 '{PRIV_NAMES[pt]}' (강화)")
                        print(f"  결과:\n{to_cpu(self.matrices[winner])}")
                    else:
                        print(f"[AI 특권] 전체 타팀에 '{PRIV_NAMES[pt]}' (약화)")
                elif winner == 1:
                    self._human_choose_privilege(winner)
                else:
                    self.agent_random_privilege(winner)
                    print("[Bot 특권] 랜덤 특권 적용")

        print("\n" + "="*50)
        print(" [최종 결과]")
        winners = self.get_final_winner()
        for i in range(self.num_players):
            label  = "AI(T1)" if i == 0 else "YOU(T2)" if i == 1 else f"Bot(T{i+1})"
            det    = self.get_score(i, mode="det")
            ax_sum = self.get_score(i, mode="ax")
            status = "WINNER" if i in winners else "SURVIVED" if det != 0 else "ELIMINATED"
            print(f" {label}: Ax={ax_sum:.1f}  det={det}  [{status}]")
        print("="*50)
        labels = {0: "AI (Team 1)", 1: "YOU (Team 2)"}
        if not winners:
            print(" 결과: 전원 탈락 (역행렬 존재 조 없음)")
        elif len(winners) > 1:
            names = [labels.get(w, f'Bot (Team {w+1})') for w in winners]
            print(f" 공동 우승: {', '.join(names)}")
        else:
            print(f" 최종 승자: {labels.get(winners[0], f'Bot (Team {winners[0]+1})')}")


if __name__ == "__main__":
    game = MatrixGame()
    print("=== Evo-Matrix AI Engine v3.0 (MLP + League) ===")
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
