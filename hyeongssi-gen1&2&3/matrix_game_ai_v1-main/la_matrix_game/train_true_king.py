import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

if __name__ == "__main__":
    print("🚀 [진짜 규칙 적용] 5세대 암살자 재학습(파인 튜닝)을 시작합니다.")
    
    # 1. 5세대 모델을 적군 풀에 넣어서, 자기 자신과 피 튀기게 싸우도록 세팅
    assassin_model = PPO.load("la_matrix_final_assassin")
    opponent_pool = [assassin_model]
    
    # 2. 아까 규칙을 수정한 훈련장 불러오기
    env = AdvancedSelfPlayEnv(opponent_models=opponent_pool)
    
    # 3. 5세대 모델의 뇌를 이 환경에 다시 연결
    # 🚨 수정된 부분: verbose=1은 여기서 세팅해야 합니다!
    model = PPO.load("la_matrix_final_assassin", env=env, verbose=1)
    
    # 4. 딱 20만 번만 광역 지령에 적응하도록 추가 학습
    print("⚔️ 훈련 시작! (20만 번, 약 5~10분 소요)")
    # 🚨 수정된 부분: learn() 함수 안에는 횟수만 적습니다.
    model.learn(total_timesteps=200000)
    
    # 5. 진정한 최종 모델 저장
    model.save("la_matrix_true_king")
    print("👑 광역 암살 마스터 'la_matrix_true_king.zip' 저장 완료!")