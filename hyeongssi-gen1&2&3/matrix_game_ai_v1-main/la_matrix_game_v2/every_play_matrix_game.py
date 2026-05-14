import numpy as np
import math
import time
from sb3_contrib import MaskablePPO

class MatrixGameFinal:
    def __init__(self, ai_model_path="gen3_league_master"):
        self.num_players = 6
        # 0번 조: AI, 1~5번 조: 인간 플레이어
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        
        self.current_round = 1
        self.used_cells_this_round = [set() for _ in range(self.num_players)]
        
        print("⏳ [시스템] 3세대 챔피언 AI의 뇌를 이식 중입니다...")
        try:
            self.ai_model = MaskablePPO.load(ai_model_path)
            print("✔️ [시스템] AI 로딩 완료! 인류의 건투를 빕니다.\n")
        except:
            print(f"❌ [오류] '{ai_model_path}.zip' 파일을 찾을 수 없습니다! 같은 폴더에 있는지 확인해주세요.")
            exit()

    def print_board(self):
        print("\n" + "="*50)
        print(f"📊 [ 현재 공통 벡터 X ]\n{self.x_vector.flatten()}")
        print("="*50)
        
        for i in range(self.num_players):
            name = "👑 [0번 조] 3세대 AI" if i == 0 else f"👤 [{i}번 조] 인간 플레이어"
            print(name)
            
            # 대각선을 '*'로 가려서 출력 (규칙 적용)
            disp_mat = []
            for r in range(3):
                row_str = []
                for c in range(3):
                    if r == c:
                        row_str.append(f" * ")
                    else:
                        row_str.append(f"{self.matrices[i][r, c]:2d} ")
                disp_mat.append("[" + " ".join(row_str) + "]")
            
            for line in disp_mat:
                print("  " + line)
            print("-" * 30)

    def calculate_x(self):
        for i in range(3):
            diag_sum = sum(p[i, i] for p in self.matrices)
            self.x_vector[i, 0] = math.floor(diag_sum / self.num_players)

    def get_score(self, player_idx, round_num):
        mat = self.matrices[player_idx]
        if round_num <= 2:
            return np.sum(np.dot(mat, self.x_vector))
        else:
            return int(round(np.linalg.det(mat)))

    def find_winner(self):
        print("\n" + "="*50)
        print("📊 [ 라운드 점수 집계 및 우승자 판독 ]")
        
        # 1. 모든 플레이어의 점수 계산
        scores = [(self.get_score(i, self.current_round), i) for i in range(self.num_players)]
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # 2. 점수별로 그룹화
        score_map = {}
        unique_scores = []
        for s, idx in scores:
            if s not in score_map:
                unique_scores.append(s)
                score_map[s] = []
            score_map[s].append(idx)
            
        # 3. 차점자 모색 (3순위까지) 및 중계
        for rank, s in enumerate(unique_scores[:3]):
            players = score_map[s]
            names = ["👑 3세대 AI(0번)" if p == 0 else f"👤 인간({p}번)" for p in players]
            print(f"  ▶ {rank+1}순위 점수 ({s}점) : {', '.join(names)}")
            
            if len(players) == 1:
                print(f"  >> 단독 우승자 판정 완료! 🎉 ({names[0]})")
                print("="*50)
                return players[0]
            else:
                print(f"  >> ⚠️ {len(players)}명 동점 발생! 다음 순위 차점자를 탐색합니다...")
                
        print("  >> ❌ 우승자 판별 불가 (단독 우승자 없음 또는 하위 점수 그룹 부재)")
        print("="*50)
        return None

    # --- AI 전용 함수 ---
    def get_ai_state(self, is_privilege):
        state = []
        state.extend(self.matrices[0].flatten())
        for i in range(1, self.num_players):
            enemy_mat = self.matrices[i].copy()
            np.fill_diagonal(enemy_mat, 0)
            state.extend(enemy_mat.flatten())
        state.extend(self.x_vector.flatten())
        state.append(self.current_round)
        
        # 🚨 오타 수정 완료! (is_priv -> is_privilege)
        state.append(1.0 if is_privilege else 0.0) 
        return np.array(state, dtype=np.float32)

    def get_ai_mask(self, is_privilege):
        mask = np.zeros(120, dtype=bool)
        if is_privilege:
            mask[18:120] = True
        else:
            for i in range(18):
                r, c = (i // 2) // 3, (i // 2) % 3
                if (r, c) not in self.used_cells_this_round[0]:
                    mask[i] = True
        return mask

    def decode_privilege(self, act_idx):
        p_idx = act_idx - 18
        t_type = 1 if p_idx % 2 == 0 else 2
        p_idx //= 2
        if p_idx < 3: return 1, t_type, [(0,1), (0,2), (1,2)][p_idx]
        p_idx -= 3
        if p_idx < 3: return 2, t_type, [(0,1), (0,2), (1,2)][p_idx]
        p_idx -= 3
        if p_idx < 9: return 3, t_type, (p_idx // 3, p_idx % 3)
        p_idx -= 9
        pairs = [(i, j) for i in range(9) for j in range(i+1, 9)]
        return 4, t_type, pairs[p_idx]

    def apply_privilege(self, winner_idx, cmd, t_type, args):
        targets = [winner_idx] if t_type == 1 else [i for i in range(self.num_players) if i != winner_idx]
        target_str = "자신" if t_type == 1 else "나머지 전원"
        
        print(f"\n🚨 [지령 발동] 대상: {target_str}")
        if cmd == 1:
            print(f"  👉 {args[0]+1}열과 {args[1]+1}열을 교환합니다.") # 출력 시 +1 보정
            for t in targets: self.matrices[t][:, list(args)] = self.matrices[t][:, reversed(list(args))]
        elif cmd == 2:
            print(f"  👉 {args[0]+1}행과 {args[1]+1}행을 교환합니다.") # 출력 시 +1 보정
            for t in targets: self.matrices[t][list(args), :] = self.matrices[t][reversed(list(args)), :]
        elif cmd == 3:
            print(f"  👉 ({args[0]+1}, {args[1]+1}) 성분을 0으로 만듭니다.") # 출력 시 +1 보정
            for t in targets: self.matrices[t][args[0], args[1]] = 0
        elif cmd == 4:
            r1, c1, r2, c2 = args[0]//3, args[0]%3, args[1]//3, args[1]%3
            print(f"  👉 ({r1+1}, {c1+1}) 성분과 ({r2+1}, {c2+1}) 성분을 교환합니다.") # 출력 시 +1 보정
            for t in targets:
                self.matrices[t][r1, c1], self.matrices[t][r2, c2] = self.matrices[t][r2, c2], self.matrices[t][r1, c1]

    # --- 인간 입력 로직 (1, 2, 3 입력 대응) ---
    def human_normal_turn(self, player_idx):
        while True:
            try:
                print(f"\n👤 [{player_idx}번 조] 이번 라운드 잔여 행동: {self.current_round - len(self.used_cells_this_round[player_idx])}회")
                user_input = input("위치를 입력하세요 (형식: 행 열 값(1또는-1) / 예: 1 2 -1): ")
                r_in, c_in, v = map(int, user_input.strip().split())
                
                # 사용자는 1, 2, 3 중 하나를 입력해야 함
                if not (1 <= r_in <= 3 and 1 <= c_in <= 3):
                    print("행과 열은 1, 2, 3 중 하나여야 합니다.")
                    continue
                if v not in [1, -1]:
                    print("값은 1 또는 -1만 가능합니다.")
                    continue
                
                # 내부 연산을 위해 1을 빼줌 (1,2,3 -> 0,1,2)
                r, c = r_in - 1, c_in - 1
                
                if (r, c) in self.used_cells_this_round[player_idx]:
                    print("이번 라운드에 이미 조작한 칸입니다! 다른 칸을 선택하세요.")
                    continue
                
                self.matrices[player_idx][r, c] += v
                self.used_cells_this_round[player_idx].add((r, c))
                print(f"✔️ [{player_idx}번 조] ({r_in}, {c_in}) 위치에 {v} 적용 완료!")
                break
            except ValueError:
                print("입력 형식이 잘못되었습니다. 띄어쓰기로 구분하여 3개의 숫자를 입력해주세요.")

    def human_privilege_turn(self, player_idx):
        print(f"\n🎉 축하합니다! 👤 [{player_idx}번 조]가 이번 라운드 단독 우승을 차지했습니다!")
        while True:
            try:
                t_type = int(input("특권 대상을 선택하세요 (1: 나 자신, 2: 나 제외 전원): "))
                if t_type not in [1, 2]: continue
                
                print("\n[특권 종류]")
                print("1. 두 열 교환")
                print("2. 두 행 교환")
                print("3. 특정 성분 0으로 만들기")
                print("4. 두 성분 교환")
                cmd = int(input("특권 번호를 선택하세요 (1~4): "))
                
                # 특권 입력도 1~3 기준으로 받아 내부에서 1을 빼줍니다.
                if cmd == 1:
                    c1, c2 = map(int, input("교환할 두 열 번호를 입력하세요 (1~3) (예: 1 2): ").split())
                    args = (c1 - 1, c2 - 1)
                elif cmd == 2:
                    r1, r2 = map(int, input("교환할 두 행 번호를 입력하세요 (1~3) (예: 2 3): ").split())
                    args = (r1 - 1, r2 - 1)
                elif cmd == 3:
                    r1, c1 = map(int, input("0으로 만들 행 열 번호를 입력하세요 (1~3) (예: 2 2): ").split())
                    args = (r1 - 1, c1 - 1)
                elif cmd == 4:
                    print("교환할 두 성분의 위치를 차례대로 입력하세요 (1~3).")
                    r1, c1, r2, c2 = map(int, input("형식: 행1 열1 행2 열2 (예: 1 1 3 3): ").split())
                    idx1 = (r1 - 1) * 3 + (c1 - 1)
                    idx2 = (r2 - 1) * 3 + (c2 - 1)
                    args = (idx1, idx2)
                else:
                    continue
                    
                self.apply_privilege(player_idx, cmd, t_type, args)
                break
            except Exception as e:
                print("입력이 잘못되었습니다. 안내된 형식에 맞게 다시 시도해주세요.")

    # --- 메인 게임 루프 ---
    # --- 메인 게임 루프 ---
    def play(self):
        print("⚔️ 인간 vs 3세대 인공지능 - 데스매치 게임을 시작합니다 ⚔️")
        
        for rnd in range(1, 6):
            self.current_round = rnd
            print(f"\n{'='*20} 🏁 [ {rnd} 라운드 시작 ] {'='*20}")
            
            for turn in range(rnd):
                # ✨ 헷갈림 방지: 현재 몇 번째 턴인지 명확하게 표시 ✨
                print(f"\n🔔 [ {rnd} 라운드 진행 중 : {turn + 1} / {rnd} 번째 턴 ] 🔔")
                self.print_board()
                
                # AI 턴
                obs = self.get_ai_state(False)
                mask = self.get_ai_mask(False)
                action, _ = self.ai_model.predict(obs, action_masks=mask, deterministic=True)
                act_idx = int(action)
                r, c, v = (act_idx // 2) // 3, (act_idx // 2) % 3, 1 if act_idx % 2 == 0 else -1
                
                self.matrices[0][r, c] += v
                self.used_cells_this_round[0].add((r, c))
                print(f"\n🤖 [AI 턴 진행] 👑 3세대 AI가 움직였습니다. (위치는 블라인드)")
                time.sleep(1)
                
                # 인간 턴
                for p in range(1, 6):
                    self.human_normal_turn(p)

            # 라운드 종료 처리
            self.calculate_x()
            for p in range(self.num_players):
                self.used_cells_this_round[p].clear()
                
            winner = self.find_winner()
            
            if winner == 0:
                print("\n🚨 [비상] 👑 3세대 AI가 라운드 단독 우승을 차지했습니다!")
                obs = self.get_ai_state(True)
                mask = self.get_ai_mask(True)
                action, _ = self.ai_model.predict(obs, action_masks=mask, deterministic=True)
                cmd, t_type, args = self.decode_privilege(int(action))
                self.apply_privilege(0, cmd, t_type, args)
                time.sleep(2)
            elif winner is not None:
                self.human_privilege_turn(winner)
            else:
                print("\n⚖️ 우승 조건 충족자가 없어 특권 없이 다음 라운드로 넘어갑니다.")

        # 게임 종료
        self.calculate_x()
        self.print_board()
        print("\n" + "="*50)
        print("====== ✨ 5라운드 종료! 최종 판독 ✨ ======")
        
        results = []
        for i in range(self.num_players):
            det = int(round(np.linalg.det(self.matrices[i])))
            ax_sum = np.sum(np.dot(self.matrices[i], self.x_vector))
            name = "👑 3세대 AI" if i == 0 else f"👤 {i}번 조"
            results.append({'id': i, 'name': name, 'det': det, 'ax': ax_sum, 'survive': det != 0})
            
            status = "✅ 생존" if det != 0 else "💀 탈락 (-50점)"
            print(f"{name} -> det(A): {det:2d} | Ax 합: {ax_sum:2d} | {status}")
            
        survivors = [r for r in results if r['survive']]
        if not survivors:
            print("\n전원 역행렬 파괴로 생존자가 없습니다. 멸망 엔딩입니다.")
        else:
            survivors.sort(key=lambda x: (x['ax'], x['det']), reverse=True)
            print(f"\n🏆 최종 우승: {survivors[0]['name']} !! (Ax 합: {survivors[0]['ax']})")
        print("="*50)

if __name__ == "__main__":
    game = MatrixGameFinal()
    game.play()