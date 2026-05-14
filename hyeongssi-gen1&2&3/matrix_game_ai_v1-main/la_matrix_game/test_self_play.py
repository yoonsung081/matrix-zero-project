import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# 훈련 때 사용했던 환경 구조 (테스트용)
class SelfPlayMatrixEnv(gym.Env):
    def __init__(self):
        super(SelfPlayMatrixEnv, self).__init__()
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

    def step(self, action, all_actions):
        # 모든 조의 행동을 반영
        for i in range(6):
            act = all_actions[i]
            row, col = (act // 2) // 3, (act // 2) % 3
            val = 1 if act % 2 == 0 else -1
            self.matrices[i][row][col] += val
            
        self.actions_taken_in_round += 1
        done = False
        if self.actions_taken_in_round >= self.current_round:
            self.current_round += 1
            self.actions_taken_in_round = 0
        if self.current_round > 5:
            done = True
        return done

# ---------------------------------------------------------
# 메인 테스트 루프
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 5세대 궁극의 AI 뇌 불러오기
    try:
        model = PPO.load("la_matrix_ultimate_alpha")
    except:
        print("❌ 모델 파일을 찾을 수 없습니다. 파일명이 정확한지 확인해주세요.")
        exit()

    env = SelfPlayMatrixEnv()
    obs_list = [env.reset()[0] for _ in range(6)] # 6명 초기 상태
    
    print("⚔️ [궁극의 AI 6인 리그전] 상대가 나 자신일 때, 어떤 전략을 쓸까?\n")
    
    done = False
    while not done:
        # 6명의 AI가 각자 자신의 행렬 상태를 보고 다음 수 결정
        all_actions = []
        for i in range(6):
            # 각 플레이어 i의 관점에서 관측값 생성
            current_obs = np.append(env.matrices[i].flatten(), env.current_round)
            action, _ = model.predict(current_obs, deterministic=True)
            all_actions.append(action)
        
        done = env.step(None, all_actions) # 전체 행동 반영

    # 결과 정산
    final_X = env._calculate_X()
    print("================ [ 최종 경기 결과 ] ================")
    print(f"공통 벡터 X:\n{final_X}\n")

    results = []
    for i in range(6):
        A = env.matrices[i]
        det_A = int(round(np.linalg.det(A)))
        Ax_sum = int(np.sum(np.dot(A, final_X)))
        results.append({'team': i+1, 'det': det_A, 'score': Ax_sum})
        print(f"{i+1}조 | det(A): {det_A:4d} | Ax 성분 합: {Ax_sum:4d}")

    # 우승자 판별
    survivors = [r for r in results if r['det'] != 0]
    if survivors:
        survivors.sort(key=lambda x: (x['score'], x['det']), reverse=True)
        print(f"\n🎉 우승: {survivors[0]['team']}조")
    else:
        print("\n💀 전원 탈락")

    print("\n[ 1조 AI가 완성한 행렬 ]")
    print(env.matrices[0])