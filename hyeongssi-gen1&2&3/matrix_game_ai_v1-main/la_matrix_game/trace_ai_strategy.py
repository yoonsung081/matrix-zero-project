import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

def trace_game():
    # 모델 로드
    model_5 = PPO.load("la_matrix_final_assassin")
    model_2 = PPO.load("la_matrix_advanced_gen_2")
    
    env = AdvancedSelfPlayEnv()
    models = [model_5] + [model_2] * 5
    
    obs, _ = env.reset()
    done = False
    
    print("🎬 5세대 암살자 AI의 전략 분석 생중계를 시작합니다.")
    print("-" * 50)

    while not done:
        round_num = env.current_round
        print(f"\n🔔 [제 {round_num} 라운드 시작]")
        
        # 이번 라운드에 할 수 있는 행동 횟수만큼 반복
        for turn in range(round_num):
            all_actions = []
            for i in range(6):
                # AI 시야: 행렬(9) + 라운드(1) + 등수(1)
                rank = 0 # 간략화
                curr_obs = np.append(env.matrices[i].flatten(), [env.current_round, rank])
                act, _ = models[i].predict(curr_obs, deterministic=True)
                all_actions.append(act)
                
                # 5세대 AI(0번)의 행동 기록
                if i == 0:
                    row, col = (act // 2) // 3, (act // 2) % 3
                    val = 1 if act % 2 == 0 else -1
                    print(f"👉 5세대 AI 행동: ({row}, {col}) 위치에 {val:2d} 추가")

            # 환경 업데이트 (내부적으로 지령 발생 여부 체크)
            # 수동으로 지령 로그를 찍기 위해 env 내부를 들여다봄
            prev_matrices = env.matrices.copy()
            obs, reward, done, _, _ = env.step(all_actions[0])
            
            # 지령으로 인한 변화 감지
            for i in range(6):
                if not np.array_equal(prev_matrices[i], env.matrices[i]) and turn == round_num - 1:
                    # 라운드 마지막 턴에 행렬이 변했다면 지령이 발동된 것
                    print(f"⚡ [지령 발생!] {i+1}조의 행렬이 공격받아 변경되었습니다.")

    print("\n" + "="*50)
    print("🏁 최종 게임 종료")
    X = env._calculate_X()
    print(f"최종 공통 벡터 X:\n{X.flatten()}")
    print("-" * 50)
    
    for i in range(6):
        A = env.matrices[i]
        det = int(round(np.linalg.det(A)))
        score = int(np.sum(np.dot(A, X)))
        tag = "👑 5세대" if i == 0 else f"👶 2세대 {i+1}조"
        status = "생존" if det != 0 else "💀탈락"
        print(f"{tag} | det: {det:4d} | 점수: {score:4d} | 결과: {status}")
        if i == 0:
            print(f"최종 행렬:\n{A}\n")

if __name__ == "__main__":
    trace_game()