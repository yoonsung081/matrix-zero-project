import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

def watch_microscopic_detail_fixed():
    print("🔍 [현미경 관전 모드 v2] 진짜 행동 추적 로그\n" + "="*70)
    
    gen_6 = PPO.load("la_matrix_gen_6_perfect")
    gen_5 = PPO.load("la_matrix_true_king")
    gen_2 = PPO.load("la_matrix_advanced_gen_2")
    
    tags = ["👑 6세대", "👴 5세대(A)", "👴 5세대(B)", "👶 2세대(A)", "👶 2세대(B)", "👶 2세대(C)"]
    
    # 🚨 버그 수정 1: 환경 내부에 진짜 적군 뇌를 이식합니다!
    opponents = [gen_5, gen_5, gen_2, gen_2, gen_2]
    env = AdvancedSelfPlayEnv(opponent_models=opponents)
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        round_num = env.current_round
        print(f"\n" + "▼"*25 + f" [ 제 {round_num} 라운드 ] " + "▼"*25)
        
        # 라운드 승자 예측
        X_now = env._calculate_X()
        scores = [np.sum(np.dot(env.matrices[i], X_now)) if round_num <= 2 else np.linalg.det(env.matrices[i]) for i in range(6)]
        expected_winner = np.argmax(scores)

        for turn in range(round_num):
            print(f"\n  [턴 {turn+1}] 실제 행동 결과:")
            
            # 주인공(6세대)의 행동만 예측
            curr_obs = np.append(env.matrices[0].flatten(), [env.current_round, 0])
            act_0, _ = gen_6.predict(curr_obs, deterministic=True)
            
            # 🚨 버그 수정 2: 환경이 진행되기 전 행렬 복사
            prev_mats = [m.copy() for m in env.matrices]
            
            # 환경 진행 (내부에 주입된 5, 2세대가 진짜로 행동함)
            obs, reward, done, _, _ = env.step(act_0)
            
            # 🚨 버그 수정 3: 가짜 예측이 아닌, '실제 행렬 변화'를 역추적하여 100% 일치하는 로그 출력
            for i in range(6):
                diff = env.matrices[i] - prev_mats[i]
                changes = np.argwhere(diff != 0)
                
                if turn < round_num - 1: # 일반 턴
                    if len(changes) == 1:
                        r, c = changes[0]
                        v = int(diff[r, c])
                        print(f"   {tags[i]:<12} -> ({r}, {c}) 위치에 {v:>2} 추가")
                else: # 마지막 턴 (숫자 추가 + 지령 발동이 겹치므로 통합 표기)
                    print(f"   {tags[i]:<12} -> 마지막 턴 행동 완료 (지령 처리 대기)")

            # 라운드 마지막 턴: 지령 결과 상세 발표
            if turn == round_num - 1:
                print("\n  [라운드 종료 및 지령 발동 결과]")
                print(f"  🏆 라운드 우승: {tags[expected_winner]}")
                
                for i in range(6):
                    # 지령으로 인한 복합 변화 확인
                    diff_count = np.count_nonzero(env.matrices[i] - prev_mats[i])
                    if diff_count > 1: 
                        if i == expected_winner:
                            print(f"   👉 🛡️ [자가 수복] {tags[i]}가 자기 행렬의 행/열을 교환해 방어했습니다.")
                        else:
                            print(f"   👉 💥 [광역 학살] 우승자의 공격! {tags[i]}의 행렬이 파괴되었습니다.")

                print("\n  [현재 각 조의 행렬 상태]")
                for i in range(6):
                    det = int(round(np.linalg.det(env.matrices[i])))
                    print(f"   {tags[i]} (det: {det:>3}):")
                    for row in env.matrices[i]:
                        print(f"     {row}")

    print("\n" + "="*70)
    print("🏁 최종 게임 종료")
    X = env._calculate_X()
    print(f"최종 공통 벡터 X: {X.flatten()}")

if __name__ == "__main__":
    watch_microscopic_detail_fixed()