import numpy as np
from sb3_contrib import MaskablePPO
from train_gen1_phase2 import MatrixGameEnvPhase2Masked

def test_phase2_detail():
    print("👁️ [2단계 상세 관전] 눈 뜬 1세대 AI와 바보 봇들의 라운드별 턴 중계\n")
    
    env = MatrixGameEnvPhase2Masked()
    model = MaskablePPO.load("gen1_phase2_eyes_open_masked")
    
    obs, _ = env.reset()
    done = False
    
    current_round_tracker = 1
    actions_this_round = []

    while not done:
        # AI 행동 예측 (마스킹 적용)
        current_action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=current_action_masks, deterministic=True)
        
        # 행동 디코딩 (AI가 어떤 행동을 했는지 기록)
        cell_idx = int(action) // 2
        row = cell_idx // 3
        col = cell_idx % 3
        val = 1 if int(action) % 2 == 0 else -1
        actions_this_round.append((row, col, val))
        
        # 환경 한 스텝 진행
        obs, reward, done, truncated, info = env.step(action)
        
        # 라운드가 바뀌었거나 게임이 끝났을 때 전체 조의 행렬 출력
        if env.unwrapped.current_round > current_round_tracker or done:
            print(f"\n" + "="*20 + f" 🏁 [ {current_round_tracker} 라운드 종료 ] " + "="*20)
            print(f"👉 AI가 이번 라운드에 둔 수 (행, 열, 값): {actions_this_round}")
            print(f"📊 현재 공통 벡터 X: {env.unwrapped.x_vector.flatten()}\n")
            
            # 1. AI 행렬 출력
            print("👑 [우리의 1세대 AI (0번 조)]")
            ai_mat = env.unwrapped.matrices[0]
            print(ai_mat)
            print(f"  -> 현재 det(A) = {int(round(np.linalg.det(ai_mat)))}")
            print(f"  -> 현재 Ax 합산 = {np.sum(np.dot(ai_mat, env.unwrapped.x_vector))}")
            print("-" * 40)
            
            # 2. 바보 봇들 행렬 출력
            for i in range(1, env.unwrapped.num_players):
                print(f"🤖 [바보 봇 ({i}번 조)]")
                bot_mat = env.unwrapped.matrices[i]
                print(bot_mat)
                print(f"  -> 현재 det(A) = {int(round(np.linalg.det(bot_mat)))}")
                print(f"  -> 현재 Ax 합산 = {np.sum(np.dot(bot_mat, env.unwrapped.x_vector))}")
                print("-" * 40)
            
            print("※ 2단계 환경이므로 우승자 특권(지령)은 발동하지 않고 다음 라운드로 넘어갑니다.")
            
            # 다음 라운드 준비
            current_round_tracker = env.unwrapped.current_round
            actions_this_round = []

    print("\n====== ✨ 2단계 최종 게임 종료 ✨ ======")
    print("관전이 종료되었습니다.")

if __name__ == "__main__":
    test_phase2_detail()