import numpy as np
import math
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

if __name__ == "__main__":
    # 1. 2세대(초보)와 5세대(암살자) 불러오기
    model_2 = PPO.load("la_matrix_advanced_gen_2")
    model_5 = PPO.load("la_matrix_final_assassin")

    env = AdvancedSelfPlayEnv()
    
    print("⚔️ [세대 간 대결] 2세대(꼼수 전문가) vs 5세대(암살자) 시뮬레이션 시작\n")
    
    # 1조: 5세대 암살자 / 2~6조: 2세대 모델들
    models = [model_5] + [model_2] * 5
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        all_actions = []
        for i in range(6):
            # 각 모델의 시점에서 관찰값 생성 (내 행렬 9개 + 라운드 + 내 등수)
            rank = 0 # 테스트용 단순화
            current_obs = np.append(env.matrices[i].flatten(), [env.current_round, rank])
            act, _ = models[i].predict(current_obs, deterministic=True)
            all_actions.append(act)
        
        # 주인공 AI(1조)의 행동만 step으로 전달하고 나머지는 내부 로직에서 처리되지만,
        # 정확한 대결을 위해 matrices를 직접 조작하는 시뮬레이션이 필요함 (아래는 간략화)
        obs, reward, done, _, _ = env.step(all_actions[0])

    print("================ [ 최종 대결 성적표 ] ================")
    final_X = env._calculate_X()
    for i in range(6):
        A = env.matrices[i]
        det = int(round(np.linalg.det(A)))
        score = int(np.sum(np.dot(A, final_X)))
        label = "👑 5세대 암살자" if i == 0 else f"👶 2세대 {i+1}조"
        print(f"{label} | det: {det:4d} | Ax합: {score:4d} | {'생존' if det!=0 else '💀탈락'}")