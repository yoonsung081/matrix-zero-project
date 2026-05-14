import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os

# ---------------------------------------------------------
# 1. 지령(특권) 시스템이 통합된 고급 리그 훈련장
# ---------------------------------------------------------
class AdvancedSelfPlayEnv(gym.Env):
    def __init__(self, opponent_models=[]):
        super(AdvancedSelfPlayEnv, self).__init__()
        # 행동 공간: 9개 위치 * (+1, -1) = 18가지
        self.action_space = spaces.Discrete(18)
        # 상태 공간: 내 행렬(9) + 라운드(1) + 내 점수 순위(1) = 11개 변수
        self.observation_space = spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)
        
        self.matrices = np.zeros((6, 3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.opponent_models = opponent_models

    def reset(self, seed=None):
        self.matrices = np.zeros((6, 3, 3), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # 내 행렬(9) + 라운드(1) + 현재 나의 등수(1)
        rank = self._get_current_rank()
        state = np.append(self.matrices[0].flatten(), [self.current_round, rank])
        return np.array(state, dtype=np.float32)

    def _get_current_rank(self):
        # 현재 라운드 기준 점수 순위 계산 (AI가 공격 타겟이 되지 않으려 노력하게 함)
        X = self._calculate_X()
        scores = []
        for i in range(6):
            if self.current_round <= 2:
                scores.append(np.sum(np.dot(self.matrices[i], X)))
            else:
                scores.append(np.linalg.det(self.matrices[i]))
        
        # 내 점수가 몇 번째로 높은지 (0~5위)
        sorted_scores = sorted(scores, reverse=True)
        return sorted_scores.index(scores[0])

    def _calculate_X(self):
        x1 = math.floor(np.mean([m[0][0] for m in self.matrices]))
        x2 = math.floor(np.mean([m[1][1] for m in self.matrices]))
        x3 = math.floor(np.mean([m[2][2] for m in self.matrices]))
        return np.array([[x1], [x2], [x3]])

    def _apply_privilege(self, winner_idx):
        # AI의 지령 타겟팅: 내 행렬식이 위험하면 '나(Self)', 안전하면 '나머지 5명 전원(Others)'
        my_det = np.linalg.det(self.matrices[winner_idx])
        
        # 1. 타겟과 지령 선택
        if abs(my_det) < 0.5: 
            target_indices = [winner_idx]  # 자가 수복 모드 (타겟: 나)
            command = random.choice([1, 2, 4]) # 0으로 만들기는 제외
        else:
            target_indices = [i for i in range(6) if i != winner_idx] # 광역 학살 모드 (타겟: 나머지 5명)
            command = 3 # 치명적인 0으로 만들기 지령
            
        # 2. 결정된 타겟들에게 지령 투하
        for t_idx in target_indices:
            mat = self.matrices[t_idx]
            if command == 1: # 열 교환 (1열과 3열)
                mat[:, [0, 2]] = mat[:, [2, 0]]
            elif command == 2: # 행 교환 (1행과 2행)
                mat[[0, 1], :] = mat[[1, 0], :]
            elif command == 3: # 0으로 만들기 (가장 절댓값이 큰 성분 파괴)
                r, c = np.unravel_index(np.argmax(np.abs(mat)), mat.shape)
                mat[r, c] = 0
            elif command == 4: # 두 성분 교환 ((1,1)과 (3,3))
                mat[0, 0], mat[2, 2] = mat[2, 2], mat[0, 0]

    def step(self, action):
        # 1. 내 행동 반영
        row, col = (action // 2) // 3, (action // 2) % 3
        val = 1 if action % 2 == 0 else -1
        self.matrices[0][row][col] += val
        
        # 2. 라이벌 AI 행동 반영
        for i in range(1, 6):
            if not self.opponent_models:
                self.matrices[i][random.randint(0,2)][random.randint(0,2)] += random.choice([1, -1])
            else:
                enemy_model = random.choice(self.opponent_models)
                enemy_obs = np.append(self.matrices[i].flatten(), [self.current_round, 0]) # 단순화된 등수 정보
                enemy_act, _ = enemy_model.predict(enemy_obs, deterministic=False)
                er, ec = (enemy_act // 2) // 3, (enemy_act // 2) % 3
                self.matrices[i][er][ec] += (1 if enemy_act % 2 == 0 else -1)

        self.actions_taken_in_round += 1
        reward = 0
        done = False
        
        # 3. 라운드 종료 체크 및 지령 발동
        if self.actions_taken_in_round >= self.current_round:
            # 라운드 승자 판별
            X = self._calculate_X()
            round_scores = []
            for i in range(6):
                if self.current_round <= 2:
                    round_scores.append(np.sum(np.dot(self.matrices[i], X)))
                else:
                    round_scores.append(np.linalg.det(self.matrices[i]))
            
            winner_idx = np.argmax(round_scores)
            
            # 지령 적용 (이 과정에서 행렬이 박살남)
            self._apply_privilege(winner_idx)
            
            # AI가 라운드 우승 시 보너스
            if winner_idx == 0: reward += 5
            
            self.current_round += 1
            self.actions_taken_in_round = 0
            
        # 4. 게임 최종 종료
        if self.current_round > 5:
            done = True
            final_X = self._calculate_X()
            ai_det = int(round(np.linalg.det(self.matrices[0])))
            
            if ai_det == 0:
                reward -= 50 
            else:
                survivors = []
                for i in range(6):
                    det = int(round(np.linalg.det(self.matrices[i])))
                    if det != 0:
                        Ax_sum = int(np.sum(np.dot(self.matrices[i], final_X)))
                        survivors.append({'team': i, 'ax_sum': Ax_sum, 'det': det})
                
                survivors.sort(key=lambda x: (x['ax_sum'], x['det']), reverse=True)
                if survivors and survivors[0]['team'] == 0:
                    reward += 100
                else:
                    reward += 10

        return self._get_obs(), reward, done, False, {}

# ---------------------------------------------------------
# 2. 고급 자가 대국 실행 (5세대 x 20만 번)
# ---------------------------------------------------------
if __name__ == "__main__":
    opponent_pool = []
    total_generations = 5
    steps_per_gen = 200000 
    
    print("🔥 [지령 시스템 통합] 100만 번의 피 튀기는 암살 리그를 시작합니다.")

    env = AdvancedSelfPlayEnv(opponent_models=opponent_pool)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/") 
    
    for gen in range(1, total_generations + 1):
        print(f"\n[Generation {gen}] 훈련 중...")
        model.learn(total_timesteps=steps_per_gen)
        
        path = f"la_matrix_advanced_gen_{gen}"
        model.save(path)
        opponent_pool.append(PPO.load(path))
        env.opponent_models = opponent_pool
        print(f"✅ {gen}세대 진화 완료 및 지령 전략 업데이트됨.")
        
    model.save("la_matrix_final_assassin")
    print("\n🏆 최종 모델 'la_matrix_final_assassin.zip' 저장 완료!")