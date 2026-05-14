# test_gen2_vs_gen1.py
import numpy as np
from sb3_contrib import MaskablePPO
from train_gen2 import MatrixGameEnvGen2

def test_gen2_match():
    print("⚔️ [2세대 vs 1세대] 세대 간 데스매치 실전 관전\n")
    
    # 환경 로드 (이 안에서 자동으로 1세대 5마리가 소환됨)
    env = MatrixGameEnvGen2()
    
    # 과적합(?)을 겪고 돌아온 2세대 AI 로드
    model_gen2 = MaskablePPO.load("gen2_self_play_master")
    
    obs, _ = env.reset()
    done = False
    current_round_tracker = 1

    print(f"========== 🏁 [ 1 라운드 시작 ] ==========")
    
    while not done:
        # 2세대 AI 행동 예측
        current_action_masks = env.action_masks()
        action, _states = model_gen2.predict(obs, action_masks=current_action_masks, deterministic=True)
        action = int(action)
        
        # 행동 생중계
        if env.unwrapped.is_privilege_turn:
            cmd, target_type, args = env.unwrapped._decode_privilege_action(action)
            target_str = "자신(1)" if target_type == 1 else "나 제외 전원(2)"
            print(f"\n🚨 [2세대 AI 특권 발동!] 지령 {cmd}번 사용! (대상: {target_str})")
        else:
            r, c, v = env.unwrapped._decode_normal_action(action)
            print(f"▶ 2세대 AI 턴: {r}행 {c}열에 {v} 더하기")

        # 스텝 진행 (여기서 1세대 봇들도 함께 움직임)
        obs, reward, done, truncated, info = env.step(action)
        
        if not done and env.unwrapped.current_round > current_round_tracker:
            print(f"\n📊 현재 공통 벡터 X: {env.unwrapped.x_vector.flatten()}")
            
            print("\n🤖 [2세대 AI (0번 조 - 도전자)]")
            gen2_mat = env.unwrapped.matrices[0]
            print(gen2_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(gen2_mat)))}, Ax: {np.sum(np.dot(gen2_mat, env.unwrapped.x_vector))}")
            
            print("\n👹 [1세대 완전체 AI (1번 조 - 적군)]")
            gen1_mat = env.unwrapped.matrices[1]
            print(gen1_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(gen1_mat)))}, Ax: {np.sum(np.dot(gen1_mat, env.unwrapped.x_vector))}")
            
            print(f"\n========== 🏁 [ {env.unwrapped.current_round} 라운드 시작 ] ==========")
            current_round_tracker = env.unwrapped.current_round

    print("\n====== ✨ 최종 데스매치 종료 ✨ ======")
    print(f"🏆 2세대 AI 최종 획득 점수: {reward} 점")

if __name__ == "__main__":
    test_gen2_match()