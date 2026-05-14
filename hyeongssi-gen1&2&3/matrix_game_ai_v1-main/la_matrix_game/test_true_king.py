import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

def watch_true_king():
    print("👑 [최종 보스전] 진정한 광역 암살 마스터의 양민 학살 쇼\n" + "="*55)
    
    env = AdvancedSelfPlayEnv()
    
    # 1조: 진정한 왕 (방금 만든 최종 모델)
    # 2~6조: 2세대 초보 모델 (광역 룰을 모르는 불쌍한 녀석들)
    king = PPO.load("la_matrix_true_king")
    noob = PPO.load("la_matrix_advanced_gen_2")
    
    models = [king] + [noob] * 5
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        round_num = env.current_round
        print(f"\n🔔 [제 {round_num} 라운드]")
        
        # 라운드 승자 예측용 계산
        X_now = env._calculate_X()
        scores = [np.sum(np.dot(env.matrices[i], X_now)) if round_num <= 2 else np.linalg.det(env.matrices[i]) for i in range(6)]
        expected_winner = np.argmax(scores)

        for turn in range(round_num):
            all_actions = []
            for i in range(6):
                curr_obs = np.append(env.matrices[i].flatten(), [env.current_round, 0])
                act, _ = models[i].predict(curr_obs, deterministic=True)
                all_actions.append(act)
                
                if i == 0:
                    r, c = (act // 2) // 3, (act // 2) % 3
                    v = 1 if act % 2 == 0 else -1
                    print(f"   턴 {turn+1}: 👑 왕의 수 -> ({r}, {c})에 {v:2d} 추가")

            # 상태 저장 후 행동 반영
            prev_mats = [m.copy() for m in env.matrices]
            obs, reward, done, _, _ = env.step(all_actions[0])
            
            # 라운드 마지막 턴: 지령 판별 중계
            if turn == round_num - 1:
                print(f"   🏆 라운드 {round_num} 우승: {expected_winner + 1}조")
                for i in range(6):
                    # 정상적인 내 턴(+1/-1) 외에 더 변했는지 확인
                    diff_count = np.count_nonzero(env.matrices[i] - prev_mats[i])
                    if diff_count > 1: 
                        if i == expected_winner:
                            print(f"      🛡️ [자가 수복] {i+1}조가 방어막을 전개했습니다.")
                        else:
                            print(f"      💥 [광역 학살] {i+1}조의 행렬이 박살 났습니다!")

    print("\n" + "="*55)
    print("🏁 최종 결과")
    X = env._calculate_X()
    print(f"최종 X 벡터: {X.flatten()}")
    print("-" * 55)
    
    for i in range(6):
        A = env.matrices[i]
        det = int(round(np.linalg.det(A)))
        score = int(np.sum(np.dot(A, X)))
        tag = "👑 왕 (True King) " if i == 0 else f"👶 초보 (2세대) {i+1}조"
        status = "생존" if det != 0 else "💀탈락"
        
        print(f"{tag} | det: {det:4d} | 점수: {score:4d} | 결과: {status}")
        if i == 0:
            print(f"   [왕의 최종 행렬]\n{A}\n")

if __name__ == "__main__":
    watch_true_king()