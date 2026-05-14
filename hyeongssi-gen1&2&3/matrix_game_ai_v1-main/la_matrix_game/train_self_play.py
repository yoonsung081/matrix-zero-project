import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

# ---------------------------------------------------------
# 1. 자가 대국(Self-Play) 전용 리그 훈련장
# ---------------------------------------------------------
class SelfPlayMatrixEnv(gym.Env):
    def __init__(self, opponent_models=[]):
        super(SelfPlayMatrixEnv, self).__init__()
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(10,), dtype=np.float32)
        self.matrices = np.zeros((6, 3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        
        # 👑 과거 세대의 AI 모델(적군)들을 저장하는 리스트
        self.opponent_models = opponent_models

    def reset(self, seed=None):
        self.matrices = np.zeros((6, 3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        state = np.append(self.matrices[0].flatten(), self.current_round)
        return np.array(state, dtype=np.float32), {}

    def _calculate_X(self):
        x1 = math.floor(np.mean([m[0][0] for m in self.matrices]))
        x2 = math.floor(np.mean([m[1][1] for m in self.matrices]))
        x3 = math.floor(np.mean([m[2][2] for m in self.matrices]))
        return np.array([[x1], [x2], [x3]])

    def step(self, action):
        # [1] 주인공 AI의 행동 (0번 조)
        row, col = (action // 2) // 3, (action // 2) % 3
        val = 1 if action % 2 == 0 else -1
        self.matrices[0][row][col] += val
        
        # [2] 5명의 라이벌 AI 행동 (1~5번 조)
        for i in range(1, 6):
            if not self.opponent_models:
                # 1세대 훈련 중이라 과거 모델이 없으면 랜덤 봇으로 작동
                r, c = random.randint(0, 2), random.randint(0, 2)
                v = random.choice([1, -1])
                self.matrices[i][r][c] += v
            else:
                # 과거 세대 AI 중 하나를 무작위로 뽑아 라이벌로 참전시킴!
                enemy_model = random.choice(self.opponent_models)
                
                # 라이벌 AI의 시야(관측값) 생성 후 행동 예측
                enemy_obs = np.append(self.matrices[i].flatten(), self.current_round)
                
                # 라이벌도 약간의 변칙(deterministic=False)을 주어 다양하게 공격하도록 함
                enemy_action, _ = enemy_model.predict(enemy_obs, deterministic=False)
                
                er, ec = (enemy_action // 2) // 3, (enemy_action // 2) % 3
                ev = 1 if enemy_action % 2 == 0 else -1
                self.matrices[i][er][ec] += ev
                
        self.actions_taken_in_round += 1
        reward = 0
        done = False
        
        # 라운드 및 보상 계산 로직 (기존과 동일)
        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0
            
        if self.current_round > 5:
            done = True
            final_X = self._calculate_X()
            ai_det = int(round(np.linalg.det(self.matrices[0])))
            
            if ai_det == 0:
                reward = -50 
            else:
                survivors = []
                for i in range(6):
                    det = int(round(np.linalg.det(self.matrices[i])))
                    if det != 0:
                        Ax_sum = int(np.sum(np.dot(self.matrices[i], final_X)))
                        survivors.append({'team': i, 'ax_sum': Ax_sum, 'det': det})
                
                survivors.sort(key=lambda x: (x['ax_sum'], x['det']), reverse=True)
                winner_idx = survivors[0]['team']
                
                # 라이벌 AI들을 꺾고 우승하면 +100점!
                if winner_idx == 0:
                    reward = 100
                else:
                    reward = 10

        next_state = np.append(self.matrices[0].flatten(), self.current_round)
        return np.array(next_state, dtype=np.float32), reward, done, False, {}

# ---------------------------------------------------------
# 2. 끝없는 진화: 세대별(Generation) 반복 학습 루프
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🧬 알파고 스타일 '자가 대국(Self-Play) 진화 시스템'을 가동합니다!\n")
    
    opponent_pool = [] # 과거 모델들이 모일 적군 대기실
    
    # 1세대부터 5세대까지 총 5번 진화시킵니다 (원하면 세대 수를 늘려도 됩니다)
    total_generations = 5
    steps_per_gen = 50000 
    
    # 환경과 최초의 주인공 AI 모델 생성
    env = SelfPlayMatrixEnv(opponent_models=opponent_pool)
    model = PPO("MlpPolicy", env, verbose=0)
    
    for gen in range(1, total_generations + 1):
        print(f"⚔️ [ {gen}세대 AI ] 훈련 시작... (라이벌 종류: {len(opponent_pool)}명)")
        
        # 현재 세대 학습 (5만 번의 대국)
        model.learn(total_timesteps=steps_per_gen)
        
        # 똑똑해진 현재 세대의 뇌를 파일로 저장 (예: ai_gen_1, ai_gen_2 ...)
        model_name = f"ai_gen_{gen}"
        model.save(model_name)
        
        # 방금 저장한 뇌를 불러와서 '적군 대기실(opponent_pool)'에 추가시킴
        opponent_pool.append(PPO.load(model_name))
        
        # 환경을 업데이트하여, 다음 세대는 방금 전까지 학습한 나와 싸우게 만듦
        env.opponent_models = opponent_pool
        
        print(f"✅ [ {gen}세대 AI ] 진화 완료! (이 모델은 다음 세대의 라이벌로 참전합니다)\n")
        
    # 최종 진화된 궁극의 모델 저장
    model.save("la_matrix_ultimate_alpha")
    print("🏆 모든 세대의 학습이 완료되었습니다! 궁극의 모델 'la_matrix_ultimate_alpha.zip'이 탄생했습니다.")