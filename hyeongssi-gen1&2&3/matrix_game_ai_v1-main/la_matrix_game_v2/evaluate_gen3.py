# evaluate_gen3.py
import numpy as np
import time
from sb3_contrib import MaskablePPO
from train_gen3_league import MatrixGameEnvGen3League

def evaluate_gen3(num_games=10000):
    print(f"⚔️ 3세대 챔피언의 진정한 실력 검증! 총 {num_games}판의 데스매치를 시작합니다.")
    print("⏳ 환경 및 과거 세대 뇌 로딩 중... (약 5~10분 정도 소요될 수 있습니다)\n")
    
    env = MatrixGameEnvGen3League()
    model_gen3 = MaskablePPO.load("gen3_league_master")
    
    wins = 0        # 1등 (100점)
    survivals = 0   # 생존 (10점)
    deaths = 0      # 탈락 (-50점)
    
    start_time = time.time()

    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        
        while not done:
            current_action_masks = env.action_masks()
            # deterministic=True 로 설정하여 AI가 무조건 최선의 수만 두도록 함
            action, _states = model_gen3.predict(obs, action_masks=current_action_masks, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            
        # 게임 종료 후 보상을 기준으로 승패 기록
        if reward == 100:
            wins += 1
        elif reward == 10:
            survivals += 1
        else:
            deaths += 1
            
        # 1000판마다 중간 진행 상황 출력
        if (i + 1) % 1000 == 0:
            current_win_rate = (wins / (i + 1)) * 100
            print(f"🔄 진행도: [{i + 1} / {num_games}] 판 완료 | 현재 승률: {current_win_rate:.2f}%")

    end_time = time.time()
    total_time = end_time - start_time
    
    # 최종 결과 통계 출력
    print("\n" + "="*40)
    print("👑 [ 3세대 챔피언 10,000판 최종 성적표 ] 👑")
    print("="*40)
    print(f"🏆 최종 우승 (1위)   : {wins} 판 ({wins/num_games*100:.2f}%)")
    print(f"🛡️ 단순 생존 (2~6위) : {survivals} 판 ({survivals/num_games*100:.2f}%)")
    print(f"💀 역행렬 파괴 (탈락) : {deaths} 판 ({deaths/num_games*100:.2f}%)")
    print("-" * 40)
    print(f"⏱️ 총 소요 시간: {total_time:.1f}초")
    print("="*40)

if __name__ == "__main__":
    evaluate_gen3(num_games=10000)