import numpy as np
from stable_baselines3 import PPO
from train_self_play_advanced import AdvancedSelfPlayEnv

# 1. 모델 로드
print("🤖 AI 군단을 정렬하는 중...")
try:
    models = [
        PPO.load("la_matrix_advanced_gen_2"),
        PPO.load("la_matrix_advanced_gen_3"),
        PPO.load("la_matrix_advanced_gen_4"),
        PPO.load("la_matrix_true_king"),
        PPO.load("la_matrix_gen_6_perfect")
    ]
    print("✅ 모든 세대 AI 참전 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

env = AdvancedSelfPlayEnv(opponent_models=models)

def print_masked_matrix(matrix, label):
    """대각 성분 (*) 처리 출력"""
    m_display = np.copy(matrix).astype(object).reshape(3, 3)
    m_display[0, 0] = m_display[1, 1] = m_display[2, 2] = '*'
    print(f"[{label}]")
    for row in m_display:
        print(f"  [ {'  '.join([f'{str(x):>5}' for x in row])} ]")

def print_all_matrices(env):
    """모든 플레이어의 현재 행렬 상태 출력"""
    print("\n" + "📊" + "-"*15 + " 현재 전체 행렬 상태 " + "-"*15 + "📊")
    for i in range(6):
        label = "나 (인간)" if i == 0 else f"AI {i+1}세대"
        print_masked_matrix(env.matrices[i], label)
    print("-" * 50)

def manual_privilege_v5(env):
    """내가 우승했을 때 일괄 적용 및 전체 공개"""
    print("\n" + "🔥"*10 + " YOU WIN! " + "🔥"*10)
    print("지령: 1(열교환), 2(행교환), 3(0으로만들기), 4(성분교환)")
    
    try:
        cmd = int(input("👉 지령 번호(1~4): "))
        scope = input("👉 대상 (s:나 자신 / o:나머지 전원): ").lower()
        targets = [0] if scope == 's' else [i for i in range(1, 6)]
        
        # 입력을 한 번만 받음
        args = {}
        if cmd == 1: args['c'] = list(map(int, input("   바꿀 두 열(0~2) 입력 (예: 0 2): ").split()))
        elif cmd == 2: args['r'] = list(map(int, input("   바꿀 두 행(0~2) 입력 (예: 0 1): ").split()))
        elif cmd == 3: args['pos'] = list(map(int, input("   0으로 만들 좌표(행 열) 입력 (예: 0 2): ").split()))
        elif cmd == 4: args['pos2'] = list(map(int, input("   교환할 두 좌표(r1 c1 r2 c2) 입력: ").split()))

        for t_idx in targets:
            mat = env.matrices[t_idx]
            if cmd == 1: mat[:, args['c']] = mat[:, args['c'][::-1]]
            elif cmd == 2: mat[args['r'], :] = mat[args['r'][::-1], :]
            elif cmd == 3: mat[args['pos'][0], args['pos'][1]] = 0
            elif cmd == 4:
                p = args['pos2']
                mat[p[0], p[1]], mat[p[2], p[3]] = mat[p[2], p[3]], mat[p[0], p[1]]
        
        print("\n✅ 당신의 지령이 전장에 반영되었습니다!")
        print_all_matrices(env)
            
    except Exception as e:
        print(f"⚠️ 입력 오류({e})! 지령 기회를 날렸습니다.")

# 3. 게임 루프
obs, _ = env.reset()
done = False
turn_count = 1
round_milestones = [1, 3, 6, 10, 15]

print("\n🚀 인류 vs AI 연합군: 최종 전쟁을 시작합니다.")

while not done:
    print(f"\n" + "="*20 + f" [ {turn_count} / 15 턴 ] " + "="*20)
    
    while True:
        try:
            action = int(input("👉 나의 행동 (0~17, 짝수:+, 홀수:-): "))
            if 0 <= action <= 17: break
        except: pass
        print("❌ 0~17 사이의 숫자를 입력하세요.")

    obs, reward, done, truncated, info = env.step(action)
    
    if turn_count in round_milestones:
        print(f"\n🔔 {turn_count}턴 종료: 라운드 결산")
        current_X = env._calculate_X()
        scores = [np.sum(np.dot(env.matrices[i], current_X)) for i in range(6)]
        winner_idx = np.argmax(scores)
        
        print(f"📊 공통 벡터 X: {current_X.flatten()}")
        print(f"🏆 우승: {'나 (인간)' if winner_idx == 0 else f'AI {winner_idx+1}세대'}")

        if winner_idx == 0:
            manual_privilege_v5(env)
        else:
            print(f"▶ {winner_idx+1}세대 AI가 특권을 사용했습니다.")
            # AI 지령 후에도 전체 행렬 공개
            print_all_matrices(env)

    turn_count += 1
    if done: break

# ==================== [ 최종 우승자 판정 ] ====================
print("\n" + "🏁"*10 + " 최종 결과 발표 " + "🏁"*10)
final_X = env._calculate_X()
print(f"최종 공통 벡터 X: {final_X.flatten()}\n")

results = []
for i in range(6):
    A = env.matrices[i]
    det = np.linalg.det(A)
    score = np.sum(np.dot(A, final_X))
    is_alive = abs(det) > 1e-5
    
    tag = "나 (인간)" if i == 0 else f"AI {i+1}세대"
    results.append({
        'tag': tag,
        'score': score,
        'det': det,
        'alive': is_alive
    })

# 생존자 중 점수 높은 순 정렬 (점수 같으면 det 높은 순)
survivors = [r for r in results if r['alive']]
survivors.sort(key=lambda x: (x['score'], abs(x['det'])), reverse=True)

for r in results:
    status = "생존" if r['alive'] else "💀탈락"
    print(f"{r['tag']:<10} | det: {r['det']:8.1f} | 점수: {r['score']:5.0f} | 결과: {status}")

print("\n" + "="*50)
if survivors:
    winner = survivors[0]
    print(f"👑 최종 우승자: {winner['tag']} !!!")
    print(f"   (점수: {winner['score']}, det: {winner['det']:.1f})")
else:
    print("💀 전원 탈락... 승리자가 없습니다.")
print("="*50)