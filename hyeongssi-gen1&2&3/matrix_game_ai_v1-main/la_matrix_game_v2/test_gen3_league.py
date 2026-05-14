import numpy as np
from sb3_contrib import MaskablePPO
from train_gen3_league import MatrixGameEnvGen3League

def test_gen3_champion():
    print("⚔️ [3세대 챔피언 관전] 3세대(나) vs 1세대(2명) vs 2세대(2명) vs 바보(1명)\n")
    
    env = MatrixGameEnvGen3League()
    model_gen3 = MaskablePPO.load("gen3_league_master")
    
    obs, _ = env.reset()
    done = False
    current_round_tracker = 1

    print(f"========== 🏁 [ 1 라운드 시작 ] ==========")
    
    while not done:
        current_action_masks = env.action_masks()
        action, _states = model_gen3.predict(obs, action_masks=current_action_masks, deterministic=True)
        action = int(action)
        
        if env.unwrapped.is_privilege_turn:
            cmd, target_type, args = env.unwrapped._decode_privilege_action(action)
            target_str = "자신(1)" if target_type == 1 else "나 제외 전원(2)"
            print(f"\n🚨 [3세대 챔피언 특권 발동!] 지령 {cmd}번 꽂아넣기! (대상: {target_str}, 상세: {args})")
        else:
            r, c, v = env.unwrapped._decode_normal_action(action)
            print(f"▶ 3세대 챔피언 턴: {r}행 {c}열에 {v} 더하기")

        obs, reward, done, truncated, info = env.step(action)
        
        # 🚨 수정된 부분: done이 True(게임 종료)일 때도 마지막 행렬을 무조건 출력하도록 변경!
        if env.unwrapped.current_round > current_round_tracker or done:
            print(f"\n📊 현재 공통 벡터 X: {env.unwrapped.x_vector.flatten()}")
            
            print("\n👑 [3세대 챔피언 (0번 조 - 완벽한 공방일체)]")
            gen3_mat = env.unwrapped.matrices[0]
            print(gen3_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(gen3_mat)))}, Ax: {np.sum(np.dot(gen3_mat, env.unwrapped.x_vector))}")
            
            print("\n🛡️ [1세대 고수 (3번 조 - 존버형)]")
            gen1_mat = env.unwrapped.matrices[3]
            print(gen1_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(gen1_mat)))}, Ax: {np.sum(np.dot(gen1_mat, env.unwrapped.x_vector))}")

            print("\n⚔️ [2세대 학살자 (5번 조 - 스나이퍼형)]")
            gen2_mat = env.unwrapped.matrices[5]
            print(gen2_mat)
            print(f"  -> det(A): {int(round(np.linalg.det(gen2_mat)))}, Ax: {np.sum(np.dot(gen2_mat, env.unwrapped.x_vector))}")
            
            # 게임이 안 끝났을 때만 다음 라운드 시작 문구 출력
            if not done:
                print(f"\n========== 🏁 [ {env.unwrapped.current_round} 라운드 시작 ] ==========")
            current_round_tracker = env.unwrapped.current_round

    print("\n====== ✨ 최종 리그전 종료 ✨ ======")
    print(f"🏆 3세대 챔피언 최종 획득 점수: {reward} 점")

if __name__ == "__main__":
    test_gen3_champion()