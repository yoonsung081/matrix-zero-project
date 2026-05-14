import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# 1. 학습할 때 썼던 게임 환경 그대로 가져오기
class MatrixGameEnv(gym.Env):
    def __init__(self):
        super(MatrixGameEnv, self).__init__()
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=-20, high=20, shape=(10,), dtype=np.float32)
        self.my_matrix = np.zeros((3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0

    def reset(self, seed=None):
        self.my_matrix = np.zeros((3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        state = np.append(self.my_matrix.flatten(), self.current_round)
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        row = (action // 2) // 3
        col = (action // 2) % 3
        value = 1 if action % 2 == 0 else -1
        
        self.my_matrix[row][col] += value
        self.actions_taken_in_round += 1
        
        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0
            
        reward = 0
        done = False
        
        if self.current_round > 5:
            done = True
            det_A = int(round(np.linalg.det(self.my_matrix)))
            if det_A == 0:
                reward = -50
            else:
                reward = 10
                
        next_state = np.append(self.my_matrix.flatten(), self.current_round)
        return np.array(next_state, dtype=np.float32), reward, done, False, {}

# ---------------------------------------------------------
# 2. 저장된 AI 두뇌 불러오기 및 단독 플레이 시뮬레이션
# ---------------------------------------------------------
if __name__ == "__main__":
    # 환경 세팅
    env = MatrixGameEnv()
    
    # 우리가 방금 훈련시킨 AI 두뇌 파일 불러오기!
    model = PPO.load("la_matrix_ai_brain")
    
    print("🤖 학습된 AI가 단독 플레이를 시작합니다...\n")
    
    # 게임 초기화
    obs, _ = env.reset()
    done = False
    
    # 게임이 끝날 때(5라운드 종료)까지 AI가 스스로 행동을 결정함
    while not done:
        # AI가 현재 행렬 상태(obs)를 보고 가장 점수를 높일 수 있는 행동을 예측
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
    print("--- 5라운드 종료 후 AI가 완성한 최종 행렬 ---")
    print(env.my_matrix)
    
    det_A = int(round(np.linalg.det(env.my_matrix)))
    print(f"\n최종 행렬식 det(A) : {det_A}")
    
    if det_A != 0:
        print("🎉 결과: 생존 성공! (역행렬을 완벽하게 만들어냈습니다)")
    else:
        print("💀 결과: 탈락 (역행렬이 없습니다)")