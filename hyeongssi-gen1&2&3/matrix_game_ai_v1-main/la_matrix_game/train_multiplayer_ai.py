import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------------------------------------------------
# 1. 6인 플레이어 경쟁 환경 구축 (Multiplayer Env)
# ---------------------------------------------------------
class MultiplayerMatrixGameEnv(gym.Env):
    def __init__(self):
        super(MultiplayerMatrixGameEnv, self).__init__()
        # AI의 행동 (18가지) 및 상태 관측 (내 행렬 9개 + 라운드 1개)
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(10,), dtype=np.float32)
        
        # 6개 조의 행렬을 동시에 관리 (인덱스 0: 우리 AI, 1~5: 랜덤 봇)
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
        # 6개 조의 대각 성분 평균값(내림)으로 공통 벡터 X 계산
        x1 = math.floor(np.mean([m[0][0] for m in self.matrices]))
        x2 = math.floor(np.mean([m[1][1] for m in self.matrices]))
        x3 = math.floor(np.mean([m[2][2] for m in self.matrices]))
        return np.array([[x1], [x2], [x3]])

    def step(self, action):
        # [1] 우리 AI의 행동 반영 (팀 0)
        row = (action // 2) // 3
        col = (action // 2) % 3
        value = 1 if action % 2 == 0 else -1
        self.matrices[0][row][col] += value
        
        # [2] 5명의 가상 봇(랜덤) 행동 반영 (팀 1~5)
        for i in range(1, 6):
            r, c = random.randint(0, 2), random.randint(0, 2)
            v = random.choice([1, -1])
            self.matrices[i][r][c] += v
            
        self.actions_taken_in_round += 1
        reward = 0
        done = False
        
        # [3] 라운드 진행 체크
        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0
            
        # [4] 5라운드 최종 종료 시 '경쟁형 보상' 계산
        if self.current_round > 5:
            done = True
            final_X = self._calculate_X()
            
            # AI의 행렬식 평가
            ai_det = int(round(np.linalg.det(self.matrices[0])))
            
            if ai_det == 0:
                reward = -50  # 역행렬 생성 실패 시 강력한 페널티
            else:
                # 6개 조 전체 성적 계산
                survivors = []
                for i in range(6):
                    det = int(round(np.linalg.det(self.matrices[i])))
                    if det != 0: # 역행렬이 존재하는 조만 생존
                        Ax_sum = int(np.sum(np.dot(self.matrices[i], final_X)))
                        survivors.append({'team': i, 'ax_sum': Ax_sum, 'det': det})
                
                # 우승자 가리기 (1순위: Ax 합, 2순위: det(A))
                survivors.sort(key=lambda x: (x['ax_sum'], x['det']), reverse=True)
                winner_idx = survivors[0]['team']
                
                if winner_idx == 0:
                    reward = 100  # AI가 최종 우승을 차지함!! (잭팟 보상)
                else:
                    reward = 10   # 생존은 했지만 우승은 못함 (기본 점수)

        # AI의 다음 상태 반환 (상대방 행렬은 비공개 유지)
        next_state = np.append(self.matrices[0].flatten(), self.current_round)
        return np.array(next_state, dtype=np.float32), reward, done, False, {}

# ---------------------------------------------------------
# 2. 업그레이드된 환경에서 AI 재훈련 시작
# ---------------------------------------------------------
if __name__ == "__main__":
    print("⚔️ 6인 경쟁 환경 세팅 완료! AI가 우승하는 법을 학습합니다. (약 2~3분 소요)")
    
    env = MultiplayerMatrixGameEnv()
    
    # 새로운 뇌 생성 후 학습 시작
    model = PPO("MlpPolicy", env, verbose=1)
    
    # 복잡한 상대방 변수가 생겼으므로 학습량을 20만 번으로 늘려줍니다.
    model.learn(total_timesteps=200000)
    
    # 우승 특화 AI 두뇌 저장
    model.save("la_matrix_champion_brain")
    print("🏆 훈련 완료! 우승 특화 AI 두뇌가 'la_matrix_champion_brain.zip'에 저장되었습니다.")