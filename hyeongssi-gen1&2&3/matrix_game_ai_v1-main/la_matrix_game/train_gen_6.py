import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

if __name__ == "__main__":
    print("🚀 [무결점 6세대 진화] 역대 모든 세대를 사냥할 진짜 AI를 훈련합니다.")
    
    # 1. 과거의 모든 세대(초보부터 왕까지)를 적군 풀에 모조리 집어넣습니다!
    opponent_pool = []
    try:
        # 남아있는 과거 모델들을 있는 대로 다 불러옵니다.
        opponent_pool.append(PPO.load("la_matrix_advanced_gen_2")) # 2세대 꼼수봇
        opponent_pool.append(PPO.load("la_matrix_final_assassin")) # 초기 5세대 암살자
        opponent_pool.append(PPO.load("la_matrix_true_king"))      # 5세대 진짜 왕
        print(f"✅ 총 {len(opponent_pool)}명의 역대 챔피언들이 적군으로 참전합니다.")
    except Exception as e:
        print("과거 모델을 불러오는 중 오류가 발생했습니다:", e)
    
    env = AdvancedSelfPlayEnv(opponent_models=opponent_pool)
    
    # 2. 6세대의 뇌는 5세대를 베이스로 시작합니다.
    model_6 = PPO.load("la_matrix_true_king", env=env, verbose=1)
    
    # 3. 모든 세대를 상대로 약점 없는 무결점 특훈 (20만 번)
    print("⚔️ 훈련 시작! 기본기부터 심리전까지 모두 파훼합니다...")
    model_6.learn(total_timesteps=200000)
    
    # 4. 무결점 6세대 저장
    model_6.save("la_matrix_gen_6_perfect")
    print("👑 약점 없는 완벽한 6세대 'la_matrix_gen_6_perfect' 탄생 완료!")