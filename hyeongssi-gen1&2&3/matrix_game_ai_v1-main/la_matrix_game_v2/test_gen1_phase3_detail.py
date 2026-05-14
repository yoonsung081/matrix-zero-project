# test_gen1_phase3_detail.py
import numpy as np
from sb3_contrib import MaskablePPO
from train_gen1_phase3 import MatrixGameEnvPhase3

def test_phase3_detail():
    print("👑 [3단계 상세 관전] 완전체 AI의 '특권(지령)' 사용 생중계\n")
    
    env = MatrixGameEnvPhase3()
    model = MaskablePPO.load("gen1_phase3_full_master")
    
    obs, _ = env.reset()
    done = False
    current_round_tracker = 1

    print(f"========== 🏁 [ 1 라운드 시작 ] ==========")
    
    while not done:
        # AI 행동 예측 (마스킹 적용)
        current_action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=current_action_masks, deterministic=True)
        action = int(action)
        
        # 행동 한글 생중계 로직
        if env.unwrapped.is_privilege_turn:
            # 특권(지령) 턴일 때
            cmd, target_type, args = env.unwrapped._decode_privilege_action(action)
            target_str = "자신(1)" if target_type == 1 else "나 제외 전원(2)"
            
            print(f"\n🚨 [우리의 AI가 라운드 우승을 차지했습니다! 특권 발동!] 🚨")
            if cmd == 1:
                print(f"  👉 지령 1번: {target_str}의 {args[0]}열과 {args[1]}열 교환")
            elif cmd == 2:
                print(f"  👉 지령 2번: {target_str}의 {args[0]}행과 {args[1]}행 교환")
            elif cmd == 3:
                print(f"  👉 지령 3번: {target_str}의 {args[0]}행 {args[1]}열 성분을 0으로 폭파!")
            elif cmd == 4:
                c1_idx, c2_idx = args
                r1, col1, r2, col2 = c1_idx//3, c1_idx%3, c2_idx//3, c2_idx%3
                print(f"  👉 지령 4번: {target_str}의 ({r1},{col1}) 성분과 ({r2},{col2}) 성분 교환")
            print("="*50)
            
        else:
            # 일반 턴일 때
            r, c, v = env.unwrapped._decode_normal_action(action)
            print(f"▶ AI 일반 턴: {r}행 {c}열에 {v} 더하기")

        # 환경 진행
        obs, reward, done, truncated, info = env.step(action)
        
        # 라운드가 바뀌었거나 끝났을 때의 행렬 출력
        if not done and env.unwrapped.current_round > current_round_tracker:
            print(f"\n📊 현재 공통 벡터 X: {env.unwrapped.x_vector.flatten()}")
            
            print("\n👑 [우리의 1세대 AI (0번 조)]")
            ai_mat = env.unwrapped.matrices[0]
            print(ai_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(ai_mat)))}, Ax 합산: {np.sum(np.dot(ai_mat, env.unwrapped.x_vector))}")
            
            # 바보 봇 하나만 대표로 출력 (출력량 조절을 위해 1번 봇만)
            print("\n🤖 [바보 봇 (1번 조) - 상태 확인용]")
            bot_mat = env.unwrapped.matrices[1]
            print(bot_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(bot_mat)))}, Ax 합산: {np.sum(np.dot(bot_mat, env.unwrapped.x_vector))}")
            
            print(f"\n========== 🏁 [ {env.unwrapped.current_round} 라운드 시작 ] ==========")
            current_round_tracker = env.unwrapped.current_round

    print("\n====== ✨ 3단계 최종 게임 종료 ✨ ======")
    print(f"👑 1세대 완전체 AI 최종 보상 획득량: {reward} 점")

if __name__ == "__main__":
    test_phase3_detail()