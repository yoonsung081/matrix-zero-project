import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# 1. 6인용 훈련장 그대로 가져오기
class MultiplayerMatrixGameEnv(gym.Env):
    def __init__(self):
        super(MultiplayerMatrixGameEnv, self).__init__()
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(10,), dtype=np.float32)
        self.matrices = np.zeros((6, 3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0

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
        # AI 행동 반영
        row = (action // 2) // 3
        col = (action // 2) % 3
        value = 1 if action % 2 == 0 else -1
        self.matrices[0][row][col] += value
        
        # 5명의 랜덤 봇 행동 반영
        for i in range(1, 6):
            r, c = random.randint(0, 2), random.randint(0, 2)
            v = random.choice([1, -1])
            self.matrices[i][r][c] += v
            
        self.actions_taken_in_round += 1
        done = False
        
        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0
            
        if self.current_round > 5:
            done = True
            
        next_state = np.append(self.matrices[0].flatten(), self.current_round)
        return np.array(next_state, dtype=np.float32), 0, done, False, {}

# ---------------------------------------------------------
# 2. 챔피언 AI 불러오기 및 6인 대전 시뮬레이션
# ---------------------------------------------------------
if __name__ == "__main__":
    env = MultiplayerMatrixGameEnv()
    
    # 우리가 방금 만든 우승 특화 AI 불러오기
    model = PPO.load("la_matrix_champion_brain")
    
    print("⚔️ 챔피언 AI vs 5명의 랜덤 봇 대결을 시작합니다...\n")
    
    obs, _ = env.reset()
    done = False
    
    # 5라운드 자동 진행
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        
    # 최종 결과 계산
    final_X = env._calculate_X()
    
    print("================ [ 최종 경기 결과 ] ================")
    print(f"최종 공통 벡터 X:\n{final_X}\n")
    
    survivors = []
    for i in range(6):
        A = env.matrices[i]
        det_A = int(round(np.linalg.det(A)))
        Ax_sum = int(np.sum(np.dot(A, final_X)))
        
        team_name = "👑 챔피언 AI (1조)" if i == 0 else f"🤖 랜덤 봇 ({i+1}조)"
        is_alive = "생존" if det_A != 0 else "탈락"
        
        print(f"{team_name} | {is_alive} | det(A): {det_A:3d} | Ax 성분 합: {Ax_sum:3d}")
        
        if det_A != 0:
            survivors.append({'team': i, 'ax_sum': Ax_sum, 'det': det_A})
            
    # 최종 우승자 판별
    if survivors:
        survivors.sort(key=lambda x: (x['ax_sum'], x['det']), reverse=True)
        winner_idx = survivors[0]['team']
        winner_name = "👑 챔피언 AI (1조)" if winner_idx == 0 else f"🤖 랜덤 봇 ({winner_idx+1}조)"
        print(f"\n🎉 최종 우승: {winner_name} !!")
    else:
        print("\n💀 전원 탈락 (우승자 없음)")
        
    print("\n[ 챔피언 AI가 만든 최종 행렬 ]")
    print(env.matrices[0])