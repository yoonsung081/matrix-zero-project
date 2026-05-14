import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces

# ★ 기본 PPO 대신 sb3_contrib의 MaskablePPO를 불러옵니다.
from sb3_contrib import MaskablePPO 

class MatrixGameEnvPhase2Masked(gym.Env):
    def __init__(self):
        super(MatrixGameEnvPhase2Masked, self).__init__()
        self.action_space = spaces.Discrete(18)
        
        # 👁️ 시야: 내 행렬(9) + 적 행렬(45) + X벡터(3) + 라운드(1) 
        # (액션 마스킹을 쓰면 AI가 '이미 누른 칸'을 스스로 기억할 필요가 없으므로 체크리스트는 시야에서 빼도 됩니다! 시야 58로 복귀)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(58,), dtype=np.float32)
        
        self.num_players = 6
        self.matrices = []
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        
        self.used_cells_this_round = [set() for _ in range(self.num_players)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        for p in range(self.num_players):
            self.used_cells_this_round[p].clear()
        return self._get_state(), {}

    def _get_state(self):
        state = []
        state.extend(self.matrices[0].flatten())
        
        for i in range(1, self.num_players):
            enemy_mat = self.matrices[i].copy()
            np.fill_diagonal(enemy_mat, 0)
            state.extend(enemy_mat.flatten())
            
        state.extend(self.x_vector.flatten())
        state.append(self.current_round)
        
        return np.array(state, dtype=np.float32)

    def _decode_action(self, action_idx):
        cell_idx = action_idx // 2
        row = cell_idx // 3
        col = cell_idx % 3
        value = 1 if action_idx % 2 == 0 else -1
        return row, col, value

    # ★ 핵심 추가: AI의 행동 버튼을 끄고 켜는 마스킹 함수
    def action_masks(self):
        """
        MaskablePPO가 매 턴마다 호출합니다.
        18개의 행동 중 선택 가능한 것은 True, 불가능한 것은 False로 반환합니다.
        """
        mask = np.ones(18, dtype=bool) # 처음엔 18개 버튼 모두 활성화(True)
        for action_idx in range(18):
            row, col, _ = self._decode_action(action_idx)
            # 만약 이번 라운드에 이미 건드린 칸이라면?
            if (row, col) in self.used_cells_this_round[0]:
                mask[action_idx] = False # 해당 칸에 +1, -1을 하는 두 버튼 모두 비활성화(False)
        return mask

    def _calculate_x(self):
        for i in range(3):
            diag_sum = sum(p[i, i] for p in self.matrices)
            self.x_vector[i, 0] = math.floor(diag_sum / self.num_players)

    def step(self, action):
        reward = 0
        
        # 1. AI 행동 적용 (이제 룰 위반 페널티 로직이 필요 없습니다. 무조건 유효한 행동만 들어옵니다!)
        row, col, value = self._decode_action(action)
        self.matrices[0][row, col] += value
        self.used_cells_this_round[0].add((row, col))

        # 2. 바보 봇 로직 (봇들도 남은 칸 중에서만 고름)
        for p in range(1, self.num_players):
            available_cells = [c for c in range(9) if (c // 3, c % 3) not in self.used_cells_this_round[p]]
            if available_cells:
                chosen_cell = random.choice(available_cells)
                b_row, b_col = chosen_cell // 3, chosen_cell % 3
                b_value = random.choice([-1, 1])
                self.matrices[p][b_row, b_col] += b_value
                self.used_cells_this_round[p].add((b_row, b_col))

        self._calculate_x()
        self.actions_taken_in_round += 1
        
        done = False

        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0
            for p in range(self.num_players):
                self.used_cells_this_round[p].clear() # 라운드 바뀌면 버튼 다시 전체 활성화

        if self.current_round > 5:
            done = True
            results = []
            for i in range(self.num_players):
                det = int(round(np.linalg.det(self.matrices[i])))
                score = np.sum(np.dot(self.matrices[i], self.x_vector))
                results.append({'id': i, 'det': det, 'score': score, 'survive': det != 0})
            
            ai_res = results[0]
            if not ai_res['survive']:
                reward -= 50
            else:
                survivors = [r for r in results if r['survive']]
                survivors.sort(key=lambda x: (x['score'], x['det']), reverse=True)
                if survivors[0]['id'] == 0:
                    reward += 100
                else:
                    reward += 10
        
        return self._get_state(), float(reward), done, False, {}

if __name__ == "__main__":
    print("👁️ [액션 마스킹 적용] 원천 차단 기술이 적용된 2단계 훈련을 시작합니다.")
    env = MatrixGameEnvPhase2Masked()
    
    # 기본 PPO가 아닌 MaskablePPO 사용
    model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    
    # 불필요한 규칙 탐색을 안 해도 되므로 15만 번으로도 충분합니다.
    model.learn(total_timesteps=150000) 
    model.save("gen1_phase2_eyes_open_masked")
    print("🎉 훈련 완료! 궁극의 2단계 뇌 'gen1_phase2_eyes_open_masked.zip'이 생성되었습니다.")