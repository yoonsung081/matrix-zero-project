import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# 1. AI가 학습할 훈련장(Environment) 구축
class MatrixGameEnv(gym.Env):
    def __init__(self):
        super(MatrixGameEnv, self).__init__()
        # 행동 공간: 9개 위치 * (+1, -1) = 18가지 선택지
        self.action_space = spaces.Discrete(18)
        # 상태 공간: 3x3 행렬(9개 숫자) + 현재 라운드(1개) = 10개의 변수 인식
        self.observation_space = spaces.Box(low=-20, high=20, shape=(10,), dtype=np.float32)
        
        self.my_matrix = np.zeros((3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0

    def reset(self, seed=None):
        # 매 게임이 새로 시작될 때마다 행렬과 라운드 초기화
        self.my_matrix = np.zeros((3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        state = np.append(self.my_matrix.flatten(), self.current_round)
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        # AI가 선택한 행동을 게임판에 반영
        row = (action // 2) // 3
        col = (action // 2) % 3
        value = 1 if action % 2 == 0 else -1
        
        self.my_matrix[row][col] += value
        self.actions_taken_in_round += 1
        
        # 1라운드는 1번, 5라운드는 5번 행동하면 다음 라운드로
        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0
            
        # 보상 산정 (당근과 채찍)
        reward = 0
        done = False
        
        if self.current_round > 5:
            done = True # 5라운드가 끝나면 게임 종료
            det_A = int(round(np.linalg.det(self.my_matrix)))
            
            # AI를 가르치는 핵심 규칙 적용
            if det_A == 0:
                reward = -50  # 역행렬을 못 만들면 강력한 페널티
            else:
                reward = 10   # 생존(det(A)!=0) 성공 시 칭찬 점수
                
        next_state = np.append(self.my_matrix.flatten(), self.current_round)
        return np.array(next_state, dtype=np.float32), reward, done, False, {}

# ---------------------------------------------------------
# 2. PPO 알고리즘 투입 및 학습 시작
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🤖 인공신경망 AI 훈련을 시작합니다. (약 1~2분 소요)")
    
    # 우리가 만든 훈련장 생성
    env = MatrixGameEnv()
    
    # 딥러닝 모델 생성 (MlpPolicy: 다층 퍼셉트론 신경망 사용)
    model = PPO("MlpPolicy", env, verbose=1)
    
    # AI가 10만 번의 행동(Step)을 하며 스스로 룰을 깨우치게 만듦
    model.learn(total_timesteps=100000)
    
    # 똑똑해진 뇌를 파일로 저장
    model.save("la_matrix_ai_brain")
    print("🎉 훈련 완료! 똑똑해진 AI의 두뇌가 'la_matrix_ai_brain.zip' 파일로 저장되었습니다.")