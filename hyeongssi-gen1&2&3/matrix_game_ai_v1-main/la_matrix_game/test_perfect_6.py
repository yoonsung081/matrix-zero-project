import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

def watch_ultimate_allstar():
    print("🏆 [올스타전] 6세대 무결점 AI vs 5세대 왕(2명) vs 2세대 꼼수(3명)\n" + "="*65)
    
    env = AdvancedSelfPlayEnv()
    
    # 모델 로드
    gen_6_perfect = PPO.load("la_matrix_gen_6_perfect")
    gen_5_king = PPO.load("la_matrix_true_king")
    gen_2_noob = PPO.load("la_matrix_advanced_gen_2")
    
    # 1조: 6세대 / 2,3조: 5세대 / 4,5,6조: 2세대
    models = [gen_6_perfect, gen_5_king, gen_5_king, gen_2_noob, gen_2_noob, gen_2_noob]
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        round_num = env.current_round
        
        # 승자 예측
        X_now = env._calculate_X()
        scores = [np.sum(np.dot(env.matrices[i], X_now)) if round_num <= 2 else np.linalg.det(env.matrices[i]) for i in range(6)]
        expected_winner = np.argmax(scores)

        for turn in range(round_num):
            all_actions = []
            for i in range(6):
                curr_obs = np.append(env.matrices[i].flatten(), [env.current_round, 0])
                act, _ = models[i].predict(curr_obs, deterministic=True)
                all_actions.append(act)

            prev_mats = [m.copy() for m in env.matrices]
            obs, reward, done, _, _ = env.step(all_actions[0])
            
            # 지령 판별 중계
            if turn == round_num - 1:
                print(f"🔔 [제 {round_num} 라운드] 우승: {expected_winner + 1}조")
                for i in range(6):
                    diff_count = np.count_nonzero(env.matrices[i] - prev_mats[i])
                    if diff_count > 1: 
                        if i == expected_winner:
                            print(f"      🛡️ [자가 수복] {i+1}조 방어막 전개")
                        else:
                            tags = ["👑 6세대", "👴 5세대", "👴 5세대", "👶 2세대", "👶 2세대", "👶 2세대"]
                            print(f"      💥 [광역 학살] {tags[i]} {i+1}조 피격!")

    print("\n" + "="*65)
    print("🏁 최종 결과")
    X = env._calculate_X()
    print(f"최종 X 벡터: {X.flatten()}")
    print("-" * 65)
    
    tags = ["👑 6세대(Perfect)", "👴 5세대(King)", "👴 5세대(King)", "👶 2세대(Noob)", "👶 2세대(Noob)", "👶 2세대(Noob)"]
    for i in range(6):
        A = env.matrices[i]
        det = int(round(np.linalg.det(A)))
        score = int(np.sum(np.dot(A, X)))
        status = "생존" if det != 0 else "💀탈락"
        
        print(f"{tags[i]:<15} | det: {det:4d} | 점수: {score:4d} | 결과: {status}")
        if i == 0:
            print(f"   [6세대의 최종 행렬]\n{A}\n")

if __name__ == "__main__":
    watch_ultimate_allstar()