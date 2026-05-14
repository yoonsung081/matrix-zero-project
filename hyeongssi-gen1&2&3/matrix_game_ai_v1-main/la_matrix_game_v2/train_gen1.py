import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class MatrixGameEnvGen1(gym.Env):
    def __init__(self):
        super(MatrixGameEnvGen1, self).__init__()
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(13,), dtype=np.float32)
        
        self.num_players = 6
        self.matrices = []
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        return self._get_state(), {}

    def _get_state(self):
        my_matrix = self.matrices[0].flatten()
        x_vec = self.x_vector.flatten()
        state = np.concatenate([my_matrix, x_vec, [self.current_round]])
        return np.array(state, dtype=np.float32)

    def _decode_action(self, action_idx):
        cell_idx = action_idx // 2
        row = cell_idx // 3
        col = cell_idx % 3
        value = 1 if action_idx % 2 == 0 else -1
        return row, col, value

    def _calculate_x(self):
        for i in range(3):
            diag_sum = sum(p[i, i] for p in self.matrices)
            self.x_vector[i, 0] = math.floor(diag_sum / self.num_players)

    def step(self, action):
        # 1. AI 및 봇 행동 적용
        row, col, value = self._decode_action(action)
        self.matrices[0][row, col] += value

        for p in range(1, self.num_players):
            bot_action = random.randint(0, 17)
            b_row, b_col, b_value = self._decode_action(bot_action)
            self.matrices[p][b_row, b_col] += b_value

        self._calculate_x()
        self.actions_taken_in_round += 1
        
        reward = 0
        done = False

        # [참고] 1~2라운드 학습 보조를 위한 중간 점수 (선택사항, 최종 보상에 비해 작게 설정)
        if self.current_round <= 2:
            ax = np.dot(self.matrices[0], self.x_vector)
            reward += float(np.sum(ax)) * 0.1 # 최종 가중치를 해치지 않도록 작게 설정

        # 라운드 교체 로직
        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0

        # 2. 게임 종료 및 최종 보상 계산 (회원님 제안 가중치)
        if self.current_round > 5:
            done = True
            
            # 모든 조의 최종 성적 집계
            results = []
            for i in range(self.num_players):
                det = int(round(np.linalg.det(self.matrices[i])))
                # 최종 우승 기준: det != 0 중 Ax 합산 최대값
                score = np.sum(np.dot(self.matrices[i], self.x_vector))
                results.append({'id': i, 'det': det, 'score': score, 'survive': det != 0})
            
            # AI(0번 조)의 상태 확인
            ai_res = results[0]
            
            if not ai_res['survive']:
                reward = -50  # 💀 역행렬 없어 탈락
            else:
                # 생존자 중 우승자 가리기 (1순위: Ax합, 2순위: det)
                survivors = [r for r in results if r['survive']]
                survivors.sort(key=lambda x: (x['score'], x['det']), reverse=True)
                
                if survivors[0]['id'] == 0:
                    reward = 100 # 🏆 5명의 봇을 꺾고 최종 우승
                else:
                    reward = 10  # 💖 생존은 했으나 우승은 못함
        
        return self._get_state(), float(reward), done, False, {}

# ---------------------------------------------------------
# 실행부 (PPO 학습 시작)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🤖 설정된 보상 체계로 1세대 AI 훈련을 시작합니다.")
    print("가중치: 우승(+100), 생존(+10), 탈락(-50)")
    
    env = MatrixGameEnvGen1()
    # n_steps와 batch_size를 조절하여 안정성을 높임
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    
    model.learn(total_timesteps=100000)
    model.save("gen1_matrix_ai_brain_v2")
    print("🎉 훈련 완료! 가중치가 반영된 'gen1_matrix_ai_brain_v2.zip'이 생성되었습니다.")