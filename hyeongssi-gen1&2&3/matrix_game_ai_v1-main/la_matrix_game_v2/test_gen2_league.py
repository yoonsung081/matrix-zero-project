# test_gen2_league.py
import numpy as np
from sb3_contrib import MaskablePPO
from train_gen2_league import MatrixGameEnvLeague

def test_league_champion():
    print("⚔️ [2세대 챔피언 관전] 바보 봇(2명) + 1세대 고수(3명) vs 2세대 챔피언(1명)\n")
    
    env = MatrixGameEnvLeague()
    model_gen2 = MaskablePPO.load("gen2_league_master")
    
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
            print(f"\n🚨 [2세대 AI 특권 발동!] 지령 {cmd}번 꽂아넣기! (대상: {target_str}, 상세: {args})")
        else:
            r, c, v = env.unwrapped._decode_normal_action(action)
            print(f"▶ 2세대 챔피언 턴: {r}행 {c}열에 {v} 더하기")

        obs, reward, done, truncated, info = env.step(action)
        
        # 라운드별 요약 (AI와 1세대 고수 1명만 비교)
        if not done and env.unwrapped.current_round > current_round_tracker:
            print(f"\n📊 현재 공통 벡터 X: {env.unwrapped.x_vector.flatten()}")
            
            print("\n👑 [2세대 챔피언 (0번 조)]")
            gen2_mat = env.unwrapped.matrices[0]
            print(gen2_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(gen2_mat)))}, Ax: {np.sum(np.dot(gen2_mat, env.unwrapped.x_vector))}")
            
            # 1~2번은 바보봇, 3~5번이 1세대 고수입니다. 3번 조(1세대)를 출력해봅시다.
            print("\n👹 [1세대 고수 형님 (3번 조)]")
            gen1_mat = env.unwrapped.matrices[3]
            print(gen1_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(gen1_mat)))}, Ax: {np.sum(np.dot(gen1_mat, env.unwrapped.x_vector))}")
            
            print(f"\n========== 🏁 [ {env.unwrapped.current_round} 라운드 시작 ] ==========")
            current_round_tracker = env.unwrapped.current_round

    print("\n====== ✨ 최종 리그전 종료 ✨ ======")
    print(f"🏆 2세대 챔피언 최종 획득 점수: {reward} 점")

if __name__ == "__main__":
    test_league_champion()