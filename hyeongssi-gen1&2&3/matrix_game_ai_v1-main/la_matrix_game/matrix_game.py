import numpy as np
import math
import random

# 1. 6개 조 초기화 (3x3 영행렬)
matrices = np.zeros((6, 3, 3), dtype=int)

# 2. X 벡터 계산 함수
def calculate_X(matrix_list):
    x1 = math.floor(np.mean([m[0][0] for m in matrix_list]))
    x2 = math.floor(np.mean([m[1][1] for m in matrix_list]))
    x3 = math.floor(np.mean([m[2][2] for m in matrix_list]))
    return np.array([[x1], [x2], [x3]])

# 3. 승리자 특권(지령) 적용 함수
def apply_privilege(matrix, command_type, args):
    new_matrix = matrix.copy()
    if command_type == 1: # 두 열 교환
        col1, col2 = args
        new_matrix[:, [col1, col2]] = new_matrix[:, [col2, col1]]
    elif command_type == 2: # 두 행 교환
        row1, row2 = args
        new_matrix[[row1, row2], :] = new_matrix[[row2, row1], :]
    elif command_type == 3: # 한 개 성분을 0으로
        row, col = args
        new_matrix[row][col] = 0
    elif command_type == 4: # 두 성분 교환
        r1, c1, r2, c2 = args
        temp = new_matrix[r1][c1]
        new_matrix[r1][c1] = new_matrix[r2][c2]
        new_matrix[r2][c2] = temp
    return new_matrix

# ---------------------------------------------------------
# 🎲 1~5라운드 메인 시뮬레이션 (로그 중계 포함)
# ---------------------------------------------------------
print("🚀 경기 시작! 랜덤 봇들이 무작위로 행동합니다...\n")

for round_num in range(1, 6):
    print(f"================ [ {round_num} 라운드 ] ================")
    
    # [행동 페이즈] 라운드 번호만큼 각 조가 무작위로 +1 또는 -1 반영
    num_actions = round_num 
    for team_idx in range(6):
        for _ in range(num_actions):
            row, col = random.randint(0, 2), random.randint(0, 2)
            value = random.choice([1, -1])
            matrices[team_idx][row][col] += value
            
    # [계산 페이즈] 라운드 종료 시점의 공통 벡터 X 업데이트
    current_X = calculate_X(matrices)
    
    # [승자 판별 페이즈] 각 조의 점수 계산
    scores = []
    for team_idx in range(6):
        if round_num <= 2:
            # 1, 2라운드는 Ax 성분 합
            Ax = np.dot(matrices[team_idx], current_X)
            score = int(np.sum(Ax))
        else:
            # 3, 4, 5라운드는 det(A)
            score = int(round(np.linalg.det(matrices[team_idx])))
        scores.append({'team': team_idx, 'score': score})
        
    # 점수 내림차순 정렬
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # 동점자(타이브레이커) 예외 처리: 1위가 동점이면 차점자가 승리
    winner_idx = scores[0]['team']
    win_score = scores[0]['score']
    
    if scores[0]['score'] == scores[1]['score']:
        print(f"⚠️ 1위 그룹 동점 발생! (점수: {scores[0]['score']}) 규칙에 따라 차점자가 승리합니다.")
        # 차점자 찾기
        for s in scores:
            if s['score'] < scores[0]['score']:
                winner_idx = s['team']
                win_score = s['score']
                break

    print(f"👑 {round_num}라운드 승리 조: {winner_idx + 1}조 (점수: {win_score})")
    
    # [특권 발동 페이즈] 무작위 대상에게 무작위 지령 내리기
    target_idx = random.randint(0, 5) # 0~5번 조 중 무작위 하나 (자기 자신 포함)
    command_type = random.randint(1, 4)
    
    if command_type == 1:
        c1, c2 = random.sample([0, 1, 2], 2)
        matrices[target_idx] = apply_privilege(matrices[target_idx], 1, [c1, c2])
        cmd_msg = f"{c1+1}열과 {c2+1}열 교환"
    elif command_type == 2:
        r1, r2 = random.sample([0, 1, 2], 2)
        matrices[target_idx] = apply_privilege(matrices[target_idx], 2, [r1, r2])
        cmd_msg = f"{r1+1}행과 {r2+1}행 교환"
    elif command_type == 3:
        r, c = random.randint(0, 2), random.randint(0, 2)
        matrices[target_idx] = apply_privilege(matrices[target_idx], 3, [r, c])
        cmd_msg = f"({r+1}, {c+1}) 성분을 0으로 만들기"
    elif command_type == 4:
        r1, c1 = random.randint(0, 2), random.randint(0, 2)
        r2, c2 = random.randint(0, 2), random.randint(0, 2)
        matrices[target_idx] = apply_privilege(matrices[target_idx], 4, [r1, c1, r2, c2])
        cmd_msg = f"({r1+1}, {c1+1}) 성분과 ({r2+1}, {c2+1}) 성분 교환"

    print(f"🎯 특권 발동: {winner_idx + 1}조가 {target_idx + 1}조에게 지령 사용! ➡️ [{cmd_msg}]\n")

    # [로그 출력 페이즈] 라운드 종료 후 각 조의 행렬 출력
    print("📊 [각 조의 행렬 상태]")
    matrix_str = ""
    for i in range(6):
        matrix_str += f"[{i+1}조]\n"
        for row in matrices[i]:
            matrix_str += "  " + " ".join(f"{val:3d}" for val in row) + "\n"
    print(matrix_str)


# ---------------------------------------------------------
# 🏆 5라운드 종료 후 최종 결과 정산
# ---------------------------------------------------------
final_X = calculate_X(matrices)

print("================ [ 최종 결과 정산 ] ================")
print(f"최종 공통 벡터 X:\n{final_X}\n")

final_scores = []
for i in range(6):
    A = matrices[i]
    det_A = int(round(np.linalg.det(A)))
    Ax = np.dot(A, final_X)
    Ax_sum = int(np.sum(Ax))
    
    is_candidate = "O (후보)" if det_A != 0 else "X (탈락)"
    final_scores.append({'team': i, 'det': det_A, 'ax_sum': Ax_sum, 'survived': det_A != 0})
    
    print(f"{i+1}조 | 생존: {is_candidate:8} | det(A): {det_A:3d} | Ax 성분 합: {Ax_sum:3d}")

# 최종 우승자 가리기 (생존자 중 Ax 합 최대, 동점 시 det(A) 최대)
survivors = [s for s in final_scores if s['survived']]
if survivors:
    # 1순위: Ax 성분 합 내림차순, 2순위: det(A) 내림차순 정렬
    survivors.sort(key=lambda x: (x['ax_sum'], x['det']), reverse=True)
    print(f"\n🎉 최종 우승 조: {survivors[0]['team'] + 1}조 !!")
else:
    print("\n💀 모든 조의 행렬식이 0입니다. 역행렬이 존재하는 조가 없어 우승자가 없습니다.")