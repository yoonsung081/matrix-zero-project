# test_gen1_phase2.py
import numpy as np
from sb3_contrib import MaskablePPO

# 작성하셨던 2단계 훈련장 환경을 불러옵니다.
# (파일 이름이 train_gen1_phase2.py 라고 가정합니다. 다를 경우 수정해주세요)
from train_gen1_phase2 import MatrixGameEnvPhase2Masked

def test_phase2_ai():
    print("👁️ [2단계 테스트] 눈 뜬 1세대 AI의 실전 플레이 관전 (액션 마스킹 적용)\n")
    
    # 1. 훈련장과 마스킹 전용 뇌 불러오기
    env = MatrixGameEnvPhase2Masked()
    model = MaskablePPO.load("gen1_phase2_eyes_open_masked")
    
    obs, _ = env.reset()
    done = False
    
    # 2. 게임 진행
    while not done:
        # ★ 핵심: AI가 생각하기 전에 '현재 활성화된 버튼 목록'을 먼저 알려줍니다.
        current_action_masks = env.action_masks()
        
        # deterministic=True: AI가 확률적 도박을 하지 않고 가장 확신하는 최고의 수만 둡니다.
        action, _states = model.predict(obs, action_masks=current_action_masks, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        
    # 3. 결과 출력
    print("====== 🏁 2단계 최종 게임 결과 ======")
    print(f"현재 공통 벡터 X:\n{env.x_vector.flatten()}\n")
    
    for i in range(env.num_players):
        mat = env.matrices[i]
        det = int(round(np.linalg.det(mat)))
        ax = np.dot(mat, env.x_vector)
        ax_sum = np.sum(ax)
        
        if i == 0:
            print("👑 [우리의 1세대 AI (눈 뜬 상태)]")
        else:
            print(f"🤖 [바보 봇 ({i}번 조)]")
            
        print(mat)
        print(f"-> det(A) = {det}")
        print(f"-> Ax 합산 = {ax_sum}")
        print("-" * 30)

if __name__ == "__main__":
    test_phase2_ai()