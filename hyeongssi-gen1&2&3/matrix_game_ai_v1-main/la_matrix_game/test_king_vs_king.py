import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

def watch_king_vs_king():
    print("👑 [신들의 전쟁] 광역 룰을 마스터한 6인의 데스매치\n" + "="*55)
    
    env = AdvancedSelfPlayEnv()
    
    # 6명 모두 방금 재학습을 마친 최종 보스(True King) 모델을 장착!
    king_model = PPO.load("la_matrix_true_king")
    models = [king_model] * 6
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        round_num = env.current_round
        print(f"\n🔔 [제 {round_num} 라운드]")
        
        # 라운드 승자 예측
        X_now = env._calculate_X()
        scores = [np.sum(np.dot(env.matrices[i], X_now)) if round_num <= 2 else np.linalg.det(env.matrices[i]) for i in range(6)]
        expected_winner = np.argmax(scores)

        for turn in range(round_num):
            all_actions = []
            for i in range(6):
                curr_obs = np.append(env.matrices[i].flatten(), [env.current_round, 0])
                # 똑같은 뇌를 가졌기 때문에 어떻게 엇갈리는지 관찰
                act, _ = models[i].predict(curr_obs, deterministic=True)
                all_actions.append(act)
                
                if i == 0:
                    r, c = (act // 2) // 3, (act // 2) % 3
                    v = 1 if act % 2 == 0 else -1
                    print(f"   턴 {turn+1}: 1조(관찰 대상) -> ({r}, {c})에 {v:2d} 추가")

            prev_mats = [m.copy() for m in env.matrices]
            obs, reward, done, _, _ = env.step(all_actions[0])
            
            # 라운드 종료 시 지령 판별
            if turn == round_num - 1:
                print(f"   🏆 라운드 {round_num} 우승: {expected_winner + 1}조")
                for i in range(6):
                    diff_count = np.count_nonzero(env.matrices[i] - prev_mats[i])
                    if diff_count > 1: 
                        if i == expected_winner:
                            print(f"      🛡️ [자가 수복] {i+1}조가 방어막(행/열 교환)을 전개했습니다.")
                        else:
                            print(f"      💥 [광역 학살] 우승자의 공격으로 {i+1}조의 행렬이 파괴되었습니다!")

    print("\n" + "="*55)
    print("🏁 최종 결과")
    X = env._calculate_X()
    print(f"최종 X 벡터: {X.flatten()}")
    print("-" * 55)
    
    for i in range(6):
        A = env.matrices[i]
        det = int(round(np.linalg.det(A)))
        score = int(np.sum(np.dot(A, X)))
        status = "생존" if det != 0 else "💀탈락"
        
        print(f"👑 고인물 {i+1}조 | det: {det:4d} | 점수: {score:4d} | 결과: {status}")
        if i == 0:
            print(f"   [1조의 최종 행렬]\n{A}\n")

if __name__ == "__main__":
    watch_king_vs_king()