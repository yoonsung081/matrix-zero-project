print("\n[System] Loading Evo-Matrix AI Engine v2.1...")
import os
import sys
import gc # Garbage Collector for memory management

# [Fix] Conda와 venv가 혼합된 환경에서 CuPy의 경로 인식 오류 해결
# venv 환경에서 실행 중일 때 Conda 환경 변수가 존재하면 CuPy 내부 탐색 로직이 충돌할 수 있습니다.
if 'VIRTUAL_ENV' in os.environ and 'CONDA_PREFIX' in os.environ:
    os.environ.pop('CONDA_PREFIX', None)

# [Fix] OS별 CUDA 경로 인식 개선
if sys.platform != 'win32':
    if 'CUDA_PATH' not in os.environ:
        # 리눅스 표준 CUDA 설치 경로 확인
        for path in ['/usr/local/cuda', '/usr/local/cuda-13', '/usr/local/cuda-12', '/usr/local/cuda-11']:
            if os.path.exists(path):
                os.environ['CUDA_PATH'] = path
                break
else:
    # [Windows] CUDA Toolkit 표준 설치 경로 탐색
    if 'CUDA_PATH' not in os.environ:
        base_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
        if os.path.exists(base_path):
            # 설치된 버전 중 가장 높은 버전 선택
            versions = sorted([d for d in os.listdir(base_path) if d.startswith('v')], reverse=True)
            if versions:
                cuda_dir = os.path.join(base_path, versions[0])
                os.environ['CUDA_PATH'] = cuda_dir
                # bin, libnvvp 경로를 PATH에 추가하여 DLL 로드 보장
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
        print("\n[Critical Error] NumPy is not installed.")
        print("Please run: pip install numpy")
        sys.exit(1)

def to_cpu(array):
    """CuPy와 NumPy 배열 모두에서 안전하게 CPU 데이터를 가져오는 헬퍼"""
    if HAS_GPU and hasattr(array, 'get'):
        return array.get()
    return array

import numpy as np
import json
import random
from tqdm import tqdm

class MatrixGame:
    def __init__(self, training=True):
        self.num_players = 6
        self.matrices = [cp.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = cp.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.training = training
        # 가중치: [상태 특징 벡터 20] -> 18가지 액션 (9칸 * (+1/-1))
        self.weights = cp.random.randn(20, 18) * 0.01
        self.learning_rate = 0.001
        self.learner_history = [] # Self-play를 위한 이전 세대 저장

    def calculate_x(self):
        stacked_matrices = cp.stack(self.matrices)
        diagonals = cp.diagonal(stacked_matrices, axis1=1, axis2=2)
        diag_sums = cp.sum(diagonals, axis=0)
        self.x_vector = cp.floor(diag_sums / self.num_players).astype(int).reshape(3, 1)

    def get_score(self, player_idx, target_round=None, mode="auto"):
        """
        mode: "auto" (라운드별 규칙), "ax" (Ax 합), "det" (행렬식)
        """
        r = target_round if target_round else self.current_round
        matrix_f = self.matrices[player_idx].astype(cp.float32)
        
        if mode == "ax" or (mode == "auto" and r <= 2):
            ax = cp.dot(matrix_f, self.x_vector.astype(cp.float32))
            return float(to_cpu(cp.sum(ax)))
        elif mode == "det" or (mode == "auto" and r > 2):
            det = cp.linalg.det(matrix_f)
            return float(to_cpu(cp.round(det)))
        return 0.0

    def get_reward(self, player_idx):
        """
        최종 우승 조건 반영: det != 0 생존자 중 Ax 합산 1위
        """
        if self.current_round == 5:
            det = self.get_score(player_idx, mode="det")
            ax_sum = self.get_score(player_idx, mode="ax")
            
            if det == 0:
                return -1000.0 # 탈락 (Singularity)
            
            # 생존 시 Ax 점수를 기본으로 하되, 1위 가능성에 가중치
            is_winner = self.get_final_winner() == player_idx
            return (ax_sum * 10) + (5000 if is_winner else 0)
        
        # 중간 라운드는 라운드 승리 여부가 중요
        return 100.0 if self.get_round_winner() == player_idx else 0.0

    def agent_v1_random(self, p_idx):
        # Random Action
        r, c = random.randint(0,2), random.randint(0,2)
        val = random.choice([-1, 1])
        self.matrices[p_idx][r, c] += val

    def agent_v2_greedy(self, p_idx):
        # Best immediate Ax score
        best_move = (0, 0, 1)
        max_s = -float('inf')
        for r in range(3):
            for c in range(3):
                for v in [-1, 1]:
                    self.matrices[p_idx][r, c] += v
                    s = self.get_score(p_idx)
                    if s > max_s:
                        max_s = s
                        best_move = (r, c, v)
                    self.matrices[p_idx][r, c] -= v
        self.matrices[p_idx][best_move[0], best_move[1]] += best_move[2]

    def agent_learner_act(self, p_idx, weights):
        """
        상태를 입력받아 액션을 결정하고, 학습을 위해 상태와 선택된 액션 인덱스를 반환합니다.
        """
        state_vec = cp.concatenate([
            self.matrices[p_idx].flatten(), 
            self.x_vector.flatten(), 
            cp.array([self.current_round])
        ])
        
        # 특징 벡터 크기 조절 (20차원)
        feat = cp.zeros(20)
        feat[:len(state_vec)] = state_vec.astype(cp.float32)
        
        logits = cp.dot(feat, weights)
        action_idx = int(to_cpu(cp.argmax(logits)))
        
        r, c = (action_idx // 2) // 3, (action_idx // 2) % 3
        v = 1 if action_idx % 2 == 0 else -1
        self.matrices[p_idx][r, c] += v
        return feat, action_idx

    def update_weights(self, episode_memory):
        """
        에피소드 종료 후 보상을 바탕으로 가중치 업데이트 (간단한 Policy Gradient 방식)
        """
        for feat, action_idx, reward in episode_memory:
            # 간단한 경사 상승법: 보상이 높을수록 해당 액션의 확률을 높임
            # (실제 RL에서는 더 복잡한 loss를 사용하지만, 여기선 직관적인 업데이트 적용)
            gradient = cp.outer(feat, cp.eye(18)[action_idx])
            self.weights += self.learning_rate * reward * gradient
        
        # 가중치 폭주 방지 (Normalization)
        self.weights = cp.clip(self.weights, -1, 1)

    def run_simulation(self, num_episodes=30000):
        print(f"Starting Simulation: {num_episodes} episodes...")
        results_data = []

        for ep in tqdm(range(num_episodes)):
            self.reset_game()
            episode_actions = [] # (feat, action_idx, round) 저장
            
            # 매 1000판마다 현재 리너를 히스토리에 추가 (Self-play용)
            if ep % 1000 == 0:
                self.learner_history.append(self.weights.copy())

            # 상대방 구성 (Mixed-Gen)
            opponents = []
            for _ in range(5):
                gen = random.random()
                if gen < 0.3: opponents.append("V1")
                elif gen < 0.6: opponents.append("V2")
                else: opponents.append("Learner_Old")

            for r in range(1, 6):
                self.current_round = r
                # 각 플레이어 턴
                for p in range(self.num_players):
                    num_actions = r
                    for _ in range(num_actions):
                        if p == 0:  # Agent_Learner (주인공)
                            feat, a_idx = self.agent_learner_act(p, self.weights)
                            episode_actions.append((feat, a_idx, r))
                        else:
                            opp_type = opponents[p-1]
                            if opp_type == "V1": self.agent_v1_random(p)
                            elif opp_type == "V2": self.agent_v2_greedy(p)
                            else: self.agent_learner_act(p, random.choice(self.learner_history))
                self.calculate_x()
                # 라운드 승자 권한 (간소화: 1위가 랜덤하게 한 명의 행렬 요소 하나를 0으로 만듦)
                winner = self.get_round_winner()
                if winner is not None:
                    target = random.choice([i for i in range(6) if i != winner])
                    self.matrices[target][random.randint(0,2), random.randint(0,2)] = 0

            # 에피소드 종료 후 최종 보상 계산 및 학습 수행
            final_learner_reward = self.get_reward(0) # Learner (Team 0)의 최종 보상
            self.update_weights(episode_actions, final_learner_reward)

            # GPU 메모리 정리 (CuPy 사용 시)
            if HAS_GPU:
                # 매 에피소드마다 정리하면 속도가 느려지므로 500판마다 수행
                if ep % 500 == 0:
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()

            # 데이터 수집 (웹 툴용)
            if ep > num_episodes - 1000:
                results_data.append({
                    "matrix": to_cpu(self.matrices[0]).tolist(),
                    "x": to_cpu(self.x_vector).flatten().tolist(),
                    "score": self.get_score(0)
                })

        # 결과 저장
        with open("strategy_weights.json", "w") as f:
            weights_data = to_cpu(self.weights).tolist()
            json.dump({"weights": weights_data, "sample_data": results_data}, f)
        print("Training Complete. Weights saved to strategy_weights.json")

    def update_weights(self, episode_actions, final_reward):
        """
        미래 가치 할인(Discount Factor)을 적용한 가중치 업데이트
        """
        gamma = 0.95 # 미래 가치 할인율
        for feat, action_idx, r_num in episode_actions:
            # 1라운드 액션이 5라운드 결과에 미치는 영향을 할인율로 계산 (Credit Assignment)
            discount = gamma ** (5 - r_num)
            gradient = cp.outer(feat, cp.eye(18, dtype=cp.float32)[action_idx])
            self.weights += self.learning_rate * (final_reward * discount) * gradient
        
        # 가중치 폭주 방지 (Normalization)
        self.weights = cp.clip(self.weights, -1, 1)


    def reset_game(self):
        self.matrices = [cp.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = cp.zeros((3, 1), dtype=int)
        self.current_round = 1

    def get_round_winner(self):
        """
        라운드 승자 판정: 단독 1위 -> 없으면 단독 2위 -> 없으면 단독 3위
        """
        player_scores = [self.get_score(i) for i in range(self.num_players)]
        score_map = {}
        for idx, s in enumerate(player_scores):
            if s not in score_map: score_map[s] = []
            score_map[s].append(idx)
        
        sorted_scores = sorted(score_map.keys(), reverse=True)
        
        for s in sorted_scores[:3]: # 1, 2, 3위까지 탐색
            if len(score_map[s]) == 1:
                return score_map[s][0]
        return None

    def get_final_winner(self):
        """
        최종 우승자 판정: det != 0 생존자 중 Ax 합산 최고 (동점 시 abs(det) 큰 쪽)
        """
        survivors = []
        for i in range(self.num_players):
            det = self.get_score(i, mode="det")
            if det != 0:
                ax_sum = self.get_score(i, mode="ax")
                survivors.append((ax_sum, abs(det), i))
        
        if not survivors: return None
        # Ax_sum 기준 내림차순, 그다음 abs(det) 기준 내림차순 정렬
        return sorted(survivors, key=lambda x: (x[0], x[1]), reverse=True)[0][2]

    def load_weights(self):
        filename = "strategy_weights.json"
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    self.weights = cp.array(data["weights"])
                print(f"Successfully loaded weights from {filename}")
                return True
            except Exception as e:
                print(f"Error loading weights: {e}")
        return False

    def play_interactive(self):
        self.reset_game()
        print("\n" + "="*45)
        print(" [MATCH] AI (Team 1) vs YOU (Team 2) vs Bots")
        print("="*45)
        
        for r in range(1, 6):
            self.current_round = r
            print(f"\n>>> ROUND {r} <<<")
            
            for p in range(self.num_players):
                num_actions = r
                if p == 0:  # AI
                    for _ in range(num_actions):
                        self.agent_learner_act(p, self.weights)
                    print("[AI] Turn complete.")
                elif p == 1:  # Human
                    print(f"\n[YOUR TURN] Your Matrix:\n{to_cpu(self.matrices[p])}")
                    for a in range(num_actions):
                        while True:
                            try:
                                move = input(f"  Action {a+1}/{num_actions} (row col val): ").split()
                                row, col, val = map(int, move)
                                if val not in [-1, 1] or not (0 <= row <= 2 and 0 <= col <= 2):
                                    raise ValueError
                                self.matrices[p][row, col] += val
                                break
                            except:
                                print("  Invalid! Format: row(0-2) col(0-2) val(-1 or 1). Ex: 1 1 1")
                else:  # Bots
                    for _ in range(num_actions):
                        self.agent_v2_greedy(p)
            
            self.calculate_x()
            print(f"\nCommon X Vector: {to_cpu(self.x_vector).flatten()}")
            
            winner = self.get_round_winner()
            if winner is not None:
                print(f"Round Winner: Team {winner + 1}")
                target = random.choice([i for i in range(self.num_players) if i != winner])
                self.matrices[target][random.randint(0,2), random.randint(0,2)] = 0

        print("\n" + "="*45)
        print(" [FINAL STATUS]")
        winner_idx = self.get_final_winner()
        for i in range(self.num_players):
            label = "AI (T1)" if i == 0 else "YOU(T2)" if i == 1 else f"Bot(T{i+1})"
            det = self.get_score(i, mode="det")
            ax_sum = self.get_score(i, mode="ax")
            status = "WINNER" if i == winner_idx else "SURVIVED" if det != 0 else "ELIMINATED"
            print(f" {label}: Ax_Sum={ax_sum}, Det={det} [{status}]")
        print("="*45)
        if winner_idx is None: print(" Result: Everyone Eliminated (All Singular Matrices)")
        else: print(f" Final Victor: Team {winner_idx + 1}")

if __name__ == "__main__":
    game = MatrixGame()
    print("=== Evo-Matrix AI Engine Manager ===")
    print("1. Performance Test (Match against AI)")
    print("2. Training (30,000 Episodes)")
    
    choice = input("\nSelect Option (1 or 2): ")
    
    if choice == "1":
        if game.load_weights():
            game.play_interactive()
        else:
            print("\n[Error] No trained weights found. Please run Option 2 first.")
    elif choice == "2":
        game.run_simulation(30000)
    else:
        print("Invalid selection. Exiting.")