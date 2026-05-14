import numpy as np
import math

class MatrixGame:
    def __init__(self):
        self.num_players = 6
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1

    def calculate_x(self):
        # 각 조의 대각 성분 (1,1), (2,2), (3,3) 평균의 내림 계산 [cite: 3139, 3140, 3141]
        for i in range(3):
            diag_sum = sum(p[i, i] for p in self.matrices)
            self.x_vector[i, 0] = math.floor(diag_sum / self.num_players)

    def get_score(self, player_idx):
        # 1~2라운드: Ax 성분 합 / 3~5라운드: det(A) 
        matrix = self.matrices[player_idx]
        if self.current_round <= 2:
            ax = np.dot(matrix, self.x_vector)
            return np.sum(ax)
        else:
            return int(round(np.linalg.det(matrix)))

    def find_winner_with_privilege(self):
        scores = [(self.get_score(i), i) for i in range(self.num_players)]
        # 점수 기준 내림차순 정렬
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # 고유 점수 그룹화 (점수별로 몇 조가 있는지 분류)
        unique_scores = []
        score_map = {}
        for s, idx in scores:
            if s not in score_map:
                unique_scores.append(s)
                score_map[s] = []
            score_map[s].append(idx)
        
        # 1순위 (최고점): 최고 점수를 받은 사람이 단 1명일 경우 승리
        if len(unique_scores) >= 1:
            first_place_score = unique_scores[0]
            if len(score_map[first_place_score]) == 1:
                return score_map[first_place_score][0] # 우승자 조 인덱스 반환
                
        # 2순위 (차점자 1): 최고점자가 2명 이상일 경우, 다음 높은 점수 확인
        if len(unique_scores) >= 2:
            second_place_score = unique_scores[1]
            if len(score_map[second_place_score]) == 1:
                return score_map[second_place_score][0]
                
        # 3순위 (차점자 2): 차점자 1도 2명 이상일 경우, 그 다음 높은 점수 확인
        if len(unique_scores) >= 3:
            third_place_score = unique_scores[2]
            if len(score_map[third_place_score]) == 1:
                return score_map[third_place_score][0]
                
        # 무효 처리: 차점자 2까지 단독이 없거나, 하위 점수 그룹이 더 이상 없는 경우
        return None

    def display_matrices(self, hide_diagonals=True):
        print(f"\n--- [현재 라운드: {self.current_round}] 모든 조의 행렬 상태 ---")
        for i in range(self.num_players):
            print(f"[{i+1}조] (Score: {self.get_score(i)})")
            mat = self.matrices[i].astype(object)
            display_mat = mat.copy()
            if hide_diagonals:
                # 대각 성분 (1,1), (2,2), (3,3)을 *로 가림 
                for d in range(3):
                    display_mat[d, d] = '*'
            print(display_mat)
        print(f"현재 공통 벡터 X:\n{self.x_vector.flatten()}")

    def apply_privilege(self, winner_idx):
        if winner_idx is None:
            print("\n⚠️ 모든 순위에 동점자가 존재하여 이번 라운드 우승자 특권이 취소되었습니다.")
            return

        print(f"\n👑 {winner_idx+1}조가 우승자 특권을 획득했습니다!")
        print("1: 열 교환, 2: 행 교환, 3: 성분 0으로 만들기, 4: 두 성분 교환")
        
        cmd = int(input("지령 번호 선택: "))
        target_type = int(input("대상 선택 (1: 자신만, 2: 나를 제외한 전원): "))
        
        targets = [winner_idx] if target_type == 1 else [i for i in range(self.num_players) if i != winner_idx]

        # 수정된 부분: 반복문 밖에서 입력을 한 번만 받도록 변경
        if cmd == 1:
            c1, c2 = map(int, input("교환할 두 열(0-2) 입력: ").split())
            for t in targets:
                self.matrices[t][:, [c1, c2]] = self.matrices[t][:, [c2, c1]]
        elif cmd == 2:
            r1, r2 = map(int, input("교환할 두 행(0-2) 입력: ").split())
            for t in targets:
                self.matrices[t][[r1, r2], :] = self.matrices[t][[r2, r1], :]
        elif cmd == 3:
            r, c = map(int, input("0으로 만들 성분 좌표(행 열) 입력: ").split())
            for t in targets:
                self.matrices[t][r, c] = 0
        elif cmd == 4:
            r1, c1, r2, c2 = map(int, input("교환할 두 성분 좌표(r1 c1 r2 c2) 입력: ").split())
            for t in targets:
                self.matrices[t][r1, c1], self.matrices[t][r2, c2] = self.matrices[t][r2, c2], self.matrices[t][r1, c1]
    def play(self):
        for r in range(1, 6):
            self.current_round = r
            num_actions = r # 라운드 번호만큼 행동 횟수 부여 
            
            for p in range(self.num_players):
                print(f"\n[{p+1}조의 턴] 총 {num_actions}번의 수정을 진행합니다.")
                for a in range(num_actions):
                    row, col, val = map(int, input(f"  ({a+1}/{num_actions}) 행(0-2) 열(0-2) 값(+1 또는 -1) 입력: ").split())
                    self.matrices[p][row, col] += val
            
            self.calculate_x()
            self.display_matrices(hide_diagonals=True)
            
            winner = self.find_winner_with_privilege()
            self.apply_privilege(winner)
            
            print("\n✨ 특권 적용 후 결과:")
            self.display_matrices(hide_diagonals=True)
            
        print("\n--- 최종 게임 종료 ---")
        # 최종 우승자 판별: det(A) != 0 중 Ax 합 최대 [cite: 3088]
        final_results = []
        for i in range(self.num_players):
            det = int(round(np.linalg.det(self.matrices[i])))
            self.current_round = 1 # Ax 합 계산을 위해 잠시 라운드 변경
            score = self.get_score(i)
            final_results.append({'idx': i+1, 'det': det, 'score': score, 'survive': det != 0})
        
        survivors = [res for res in final_results if res['survive']]
        if not survivors:
            print("모든 조의 역행렬이 존재하지 않아 우승자가 없습니다.")
        else:
            survivors.sort(key=lambda x: (x['score'], x['det']), reverse=True)
            winner = survivors[0]
            print(f"🏆 최종 승리: {winner['idx']}조 (Ax 합: {winner['score']}, det: {winner['det']})")

if __name__ == "__main__":
    game = MatrixGame()
    game.play()