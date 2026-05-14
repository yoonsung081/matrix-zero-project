import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO

class MatrixGameEnvGen3League(gym.Env):
    def __init__(self):
        super(MatrixGameEnvGen3League, self).__init__()
        self.action_space = spaces.Discrete(120)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(59,), dtype=np.float32)
        
        self.num_players = 6
        self.matrices = []
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.used_cells_this_round = [set() for _ in range(self.num_players)]
        self.is_privilege_turn = False
        
        # 🤖 [알파스타 리그 시스템 진화] 1세대와 2세대의 뇌를 모두 불러옵니다!
        print("⏳ 과거 세대의 챔피언들을 리그로 소환합니다...")
        self.bot_brain_gen1 = MaskablePPO.load("gen1_phase3_full_master")
        self.bot_brain_gen2 = MaskablePPO.load("gen2_league_master")
        print("✔️ 1세대(존버형) & 2세대(공격형) 소환 완료!")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.is_privilege_turn = False
        for p in range(self.num_players):
            self.used_cells_this_round[p].clear()
        return self._get_state_for_player(0, self.is_privilege_turn), {}

    def _get_state_for_player(self, player_idx, is_priv):
        state = []
        state.extend(self.matrices[player_idx].flatten())
        for i in range(self.num_players):
            if i != player_idx:
                enemy_mat = self.matrices[i].copy()
                np.fill_diagonal(enemy_mat, 0)
                state.extend(enemy_mat.flatten())
        state.extend(self.x_vector.flatten())
        state.append(self.current_round)
        state.append(1.0 if is_priv else 0.0)
        return np.array(state, dtype=np.float32)

    def _get_mask_for_player(self, player_idx, is_priv):
        mask = np.zeros(120, dtype=bool)
        if is_priv:
            mask[18:120] = True
        else:
            for i in range(18):
                r, c, _ = self._decode_normal_action(i)
                if (r, c) not in self.used_cells_this_round[player_idx]:
                    mask[i] = True
        return mask

    def action_masks(self):
        return self._get_mask_for_player(0, self.is_privilege_turn)

    def _decode_normal_action(self, action_idx):
        return (action_idx // 2) // 3, (action_idx // 2) % 3, 1 if action_idx % 2 == 0 else -1

    def _decode_privilege_action(self, act_idx):
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

    def _apply_privilege(self, w_idx, cmd, t_type, args):
        targets = [w_idx] if t_type == 1 else [i for i in range(self.num_players) if i != w_idx]
        for t in targets:
            if cmd == 1: 
                c1, c2 = args
                self.matrices[t][:, [c1, c2]] = self.matrices[t][:, [c2, c1]]
            elif cmd == 2: 
                r1, r2 = args
                self.matrices[t][[r1, r2], :] = self.matrices[t][[r2, r1], :]
            elif cmd == 3: 
                self.matrices[t][args[0], args[1]] = 0
            elif cmd == 4:
                r1, c1, r2, c2 = args[0]//3, args[0]%3, args[1]//3, args[1]%3
                self.matrices[t][r1, c1], self.matrices[t][r2, c2] = self.matrices[t][r2, c2], self.matrices[t][r1, c1]

    def _calculate_x(self):
        for i in range(3):
            self.x_vector[i, 0] = math.floor(sum(p[i, i] for p in self.matrices) / self.num_players)

    def _get_score(self, p_idx):
        mat = self.matrices[p_idx]
        return np.sum(np.dot(mat, self.x_vector)) if self.current_round <= 2 else int(round(np.linalg.det(mat)))

    def _find_winner(self):
        scores = [(self._get_score(i), i) for i in range(self.num_players)]
        scores.sort(key=lambda x: x[0], reverse=True)
        unique, s_map = [], {}
        for s, idx in scores:
            if s not in s_map:
                unique.append(s); s_map[s] = []
            s_map[s].append(idx)
        for i in range(min(3, len(unique))):
            if len(s_map[unique[i]]) == 1: return s_map[unique[i]][0]
        return None

    def step(self, action):
        reward = 0
        done = False

        if self.is_privilege_turn:
            cmd, t_type, args = self._decode_privilege_action(action)
            self._apply_privilege(0, cmd, t_type, args)
            self.is_privilege_turn = False
            self.current_round += 1
            if self.current_round > 5: return self._end_game()
            return self._get_state_for_player(0, False), reward, done, False, {}

        # 1. 3세대 AI 행동 적용
        r, c, v = self._decode_normal_action(action)
        self.matrices[0][r, c] += v
        self.used_cells_this_round[0].add((r, c))

        # 2. 적군 행동 적용 (심화 리그: 1번 바보, 23번 1세대, 45번 2세대)
        for p in range(1, self.num_players):
            if p == 1: # 바보 봇 (샌드백)
                avail = [idx for idx in range(18) if (self._decode_normal_action(idx)[0], self._decode_normal_action(idx)[1]) not in self.used_cells_this_round[p]]
                if avail:
                    br, bc, bv = self._decode_normal_action(random.choice(avail))
                    self.matrices[p][br, bc] += bv
                    self.used_cells_this_round[p].add((br, bc))
            else:
                obs_p = self._get_state_for_player(p, False)
                mask_p = self._get_mask_for_player(p, False)
                
                # 조 번호에 따라 다른 뇌를 사용!
                if p in [2, 3]: # 1세대 (존버충)
                    bot_act, _ = self.bot_brain_gen1.predict(obs_p, action_masks=mask_p, deterministic=True)
                else:           # 4, 5번 조는 2세대 (학살자)
                    bot_act, _ = self.bot_brain_gen2.predict(obs_p, action_masks=mask_p, deterministic=True)
                    
                br, bc, bv = self._decode_normal_action(int(bot_act))
                self.matrices[p][br, bc] += bv
                self.used_cells_this_round[p].add((br, bc))

        self._calculate_x()
        self.actions_taken_in_round += 1

        if self.actions_taken_in_round >= self.current_round:
            self.actions_taken_in_round = 0
            for p in range(self.num_players): self.used_cells_this_round[p].clear()

            winner_idx = self._find_winner()
            
            if winner_idx == 0:
                self.is_privilege_turn = True
                return self._get_state_for_player(0, True), 0, False, False, {} 
                
            elif winner_idx is not None:
                if winner_idx == 1: # 바보 봇 우승
                    cmd, t_type, args = self._decode_privilege_action(random.randint(18, 119))
                    self._apply_privilege(winner_idx, cmd, t_type, args)
                else: 
                    obs_w = self._get_state_for_player(winner_idx, True)
                    mask_w = self._get_mask_for_player(winner_idx, True)
                    
                    if winner_idx in [2, 3]: # 1세대 우승 지령
                        priv_act, _ = self.bot_brain_gen1.predict(obs_w, action_masks=mask_w, deterministic=True)
                    else:                    # 2세대 우승 지령 (위험!)
                        priv_act, _ = self.bot_brain_gen2.predict(obs_w, action_masks=mask_w, deterministic=True)
                        
                    cmd, t_type, args = self._decode_privilege_action(int(priv_act))
                    self._apply_privilege(winner_idx, cmd, t_type, args)
                
            self.current_round += 1

        if self.current_round > 5:
            return self._end_game()

        return self._get_state_for_player(0, False), reward, done, False, {}

    def _end_game(self):
        results = []
        for i in range(self.num_players):
            det = int(round(np.linalg.det(self.matrices[i])))
            score = np.sum(np.dot(self.matrices[i], self.x_vector))
            results.append({'id': i, 'det': det, 'score': score, 'survive': det != 0})
        
        reward = -50 if not results[0]['survive'] else 10
        if results[0]['survive']:
            survs = [r for r in results if r['survive']]
            survs.sort(key=lambda x: (x['score'], x['det']), reverse=True)
            if survs[0]['id'] == 0: reward = 100
            
        return self._get_state_for_player(0, False), float(reward), True, False, {}

if __name__ == "__main__":
    print("🔥 [3세대 리그 출범] 1세대(존버)와 2세대(스나이퍼)가 모두 섞인 최악의 지옥불 리그가 시작됩니다!")
    env = MatrixGameEnvGen3League()
    
    # 🌟 2세대의 뇌를 이어받아 3세대 훈련 시작 (target_kl로 뇌 붕괴 방지)
    model = MaskablePPO.load("gen2_league_master", env=env, target_kl=0.015)
    
    # 더 험난한 리그이므로 50만 판 진행!
    model.learn(total_timesteps=500000) 
    model.save("gen3_league_master")
    print("🎉 훈련 완료! 공방의 완벽한 밸런스를 갖춘 3세대 AI(gen3_league_master)가 탄생했습니다!")

30725이형래
1분
import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO

class MatrixGameEnvLeague(gym.Env):
    def __init__(self):
        super(MatrixGameEnvLeague, self).__init__()
        self.action_space = spaces.Discrete(120)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(59,), dtype=np.float32)
        
        self.num_players = 6
        self.matrices = []
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.used_cells_this_round = [set() for _ in range(self.num_players)]
        self.is_privilege_turn = False
        
        # 🤖 [알파스타 리그 시스템] 1세대 뇌를 불러옵니다.
        print("⏳ 1세대 챔피언의 뇌를 복제하여 리그를 구성합니다...")
        self.bot_brain = MaskablePPO.load("gen1_phase3_full_master")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.is_privilege_turn = False
        for p in range(self.num_players):
            self.used_cells_this_round[p].clear()
        return self._get_state_for_player(0, self.is_privilege_turn), {}

    def _get_state_for_player(self, player_idx, is_priv):
        state = []
        state.extend(self.matrices[player_idx].flatten())
        for i in range(self.num_players):
            if i != player_idx:
                enemy_mat = self.matrices[i].copy()
                np.fill_diagonal(enemy_mat, 0)
                state.extend(enemy_mat.flatten())
        state.extend(self.x_vector.flatten())
        state.append(self.current_round)
        state.append(1.0 if is_priv else 0.0)
        return np.array(state, dtype=np.float32)

    def _get_mask_for_player(self, player_idx, is_priv):
        mask = np.zeros(120, dtype=bool)
        if is_priv:
            mask[18:120] = True
        else:
            for i in range(18):
                r, c, _ = self._decode_normal_action(i)
                if (r, c) not in self.used_cells_this_round[player_idx]:
                    mask[i] = True
        return mask

    def action_masks(self):
        return self._get_mask_for_player(0, self.is_privilege_turn)

    def _decode_normal_action(self, action_idx):
        return (action_idx // 2) // 3, (action_idx // 2) % 3, 1 if action_idx % 2 == 0 else -1

    def _decode_privilege_action(self, act_idx):
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

    def _apply_privilege(self, w_idx, cmd, t_type, args):
        targets = [w_idx] if t_type == 1 else [i for i in range(self.num_players) if i != w_idx]
        for t in targets:
            if cmd == 1: 
                c1, c2 = args
                self.matrices[t][:, [c1, c2]] = self.matrices[t][:, [c2, c1]]
            elif cmd == 2: 
                r1, r2 = args
                self.matrices[t][[r1, r2], :] = self.matrices[t][[r2, r1], :]
            elif cmd == 3: 
                self.matrices[t][args[0], args[1]] = 0
            elif cmd == 4:
                r1, c1, r2, c2 = args[0]//3, args[0]%3, args[1]//3, args[1]%3
                self.matrices[t][r1, c1], self.matrices[t][r2, c2] = self.matrices[t][r2, c2], self.matrices[t][r1, c1]

    def _calculate_x(self):
        for i in range(3):
            self.x_vector[i, 0] = math.floor(sum(p[i, i] for p in self.matrices) / self.num_players)

    def _get_score(self, p_idx):
        mat = self.matrices[p_idx]
        return np.sum(np.dot(mat, self.x_vector)) if self.current_round <= 2 else int(round(np.linalg.det(mat)))

    def _find_winner(self):
        scores = [(self._get_score(i), i) for i in range(self.num_players)]
        scores.sort(key=lambda x: x[0], reverse=True)
        unique, s_map = [], {}
        for s, idx in scores:
            if s not in s_map:
                unique.append(s); s_map[s] = []
            s_map[s].append(idx)
        for i in range(min(3, len(unique))):
            if len(s_map[unique[i]]) == 1: return s_map[unique[i]][0]
        return None

    def step(self, action):
        reward = 0
        done = False

        # [특권 턴]
        if self.is_privilege_turn:
            cmd, t_type, args = self._decode_privilege_action(action)
            self._apply_privilege(0, cmd, t_type, args)
            self.is_privilege_turn = False
            self.current_round += 1
            if self.current_round > 5: return self._end_game()
            return self._get_state_for_player(0, False), reward, done, False, {}

        # [일반 턴] 1. 2세대 AI 행동 적용
        r, c, v = self._decode_normal_action(action)
        self.matrices[0][r, c] += v
        self.used_cells_this_round[0].add((r, c))

        # 2. 적군 행동 적용 (혼합 리그: 12번은 바보 봇, 35번은 1세대 챔피언)
        for p in range(1, self.num_players):
            if p <= 2: # 샌드백 역할 (바보 봇)
                avail = [idx for idx in range(18) if (self._decode_normal_action(idx)[0], self._decode_normal_action(idx)[1]) not in self.used_cells_this_round[p]]
                if avail:
                    br, bc, bv = self._decode_normal_action(random.choice(avail))
                    self.matrices[p][br, bc] += bv
                    self.used_cells_this_round[p].add((br, bc))
            else:      # 스파링 파트너 (1세대 고수)
                obs_p = self._get_state_for_player(p, False)
                mask_p = self._get_mask_for_player(p, False)
                bot_act, _ = self.bot_brain.predict(obs_p, action_masks=mask_p, deterministic=True)
                br, bc, bv = self._decode_normal_action(int(bot_act))
                self.matrices[p][br, bc] += bv
                self.used_cells_this_round[p].add((br, bc))

        self._calculate_x()
        self.actions_taken_in_round += 1

        # 라운드 종료
        if self.actions_taken_in_round >= self.current_round:
            self.actions_taken_in_round = 0
            for p in range(self.num_players): self.used_cells_this_round[p].clear()

            winner_idx = self._find_winner()
            
            if winner_idx == 0:
                self.is_privilege_turn = True
                return self._get_state_for_player(0, True), 0, False, False, {} 
                
            elif winner_idx is not None:
                if winner_idx <= 2: # 바보 봇이 우승하면 무작위 지령
                    cmd, t_type, args = self._decode_privilege_action(random.randint(18, 119))
                    self._apply_privilege(winner_idx, cmd, t_type, args)
                else: # 1세대 고수가 우승하면 지능적 지령
                    obs_w = self._get_state_for_player(winner_idx, True)
                    mask_w = self._get_mask_for_player(winner_idx, True)
                    priv_act, _ = self.bot_brain.predict(obs_w, action_masks=mask_w, deterministic=True)
                    cmd, t_type, args = self._decode_privilege_action(int(priv_act))
                    self._apply_privilege(winner_idx, cmd, t_type, args)
                
            self.current_round += 1

        if self.current_round > 5:
            return self._end_game()

        return self._get_state_for_player(0, False), reward, done, False, {}

    def _end_game(self):
        results = []
        for i in range(self.num_players):
            det = int(round(np.linalg.det(self.matrices[i])))
            score = np.sum(np.dot(self.matrices[i], self.x_vector))
            results.append({'id': i, 'det': det, 'score': score, 'survive': det != 0})
        
        reward = -50 if not results[0]['survive'] else 10
        if results[0]['survive']:
            survs = [r for r in results if r['survive']]
            survs.sort(key=lambda x: (x['score'], x['det']), reverse=True)
            if survs[0]['id'] == 0: reward = 100
            
        return self._get_state_for_player(0, False), float(reward), True, False, {}

if __name__ == "__main__":
    print("🔥 [알파스타식 혼합 리그] 뇌 붕괴를 막기 위한 맞춤형 재활 훈련을 시작합니다!")
    env = MatrixGameEnvLeague()
    
    # 🚨 뇌 붕괴 방지용 안전벨트(target_kl=0.015) 장착!
    # 기존에 깨졌던 2세대 뇌 말고, 완벽했던 '1세대 뇌'에서 다시 출발합니다!
    model = MaskablePPO.load("gen1_phase3_full_master", env=env, target_kl=0.015)
    
    model.learn(total_timesteps=400000) 
    model.save("gen2_league_master")
    print("🎉 훈련 완료! 혼합 리그를 제패한 진짜 2세대 AI(gen2_league_master)가 탄생했습니다!")

30725이형래
2분
import numpy as np
import math
import random
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO

class MatrixGameEnvPhase3(gym.Env):
    def __init__(self):
        super(MatrixGameEnvPhase3, self).__init__()
        # 일반행동 18개 + 지령행동 102개 = 총 120개의 행동 공간
        self.action_space = spaces.Discrete(120)
        
        # 👁️ 시야: 내 행렬(9) + 적 행렬(45) + X벡터(3) + 라운드(1) + 특권 턴 여부(1) = 59
        self.observation_space = spaces.Box(low=-50, high=50, shape=(59,), dtype=np.float32)
        
        self.num_players = 6
        self.matrices = []
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.used_cells_this_round = [set() for _ in range(self.num_players)]
        
        # 특권 턴 제어 변수
        self.is_privilege_turn = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.is_privilege_turn = False
        for p in range(self.num_players):
            self.used_cells_this_round[p].clear()
        return self._get_state(), {}

    def _get_state(self):
        state = []
        state.extend(self.matrices[0].flatten())
        for i in range(1, self.num_players):
            enemy_mat = self.matrices[i].copy()
            np.fill_diagonal(enemy_mat, 0)
            state.extend(enemy_mat.flatten())
            
        state.extend(self.x_vector.flatten())
        state.append(self.current_round)
        state.append(1.0 if self.is_privilege_turn else 0.0) # 특권 턴 알리미
        return np.array(state, dtype=np.float32)

    # --- 행동 변환 (디코딩) 함수들 ---
    def _decode_normal_action(self, action_idx):
        row = (action_idx // 2) // 3
        col = (action_idx // 2) % 3
        val = 1 if action_idx % 2 == 0 else -1
        return row, col, val

    def _decode_privilege_action(self, act_idx):
        # 18 ~ 119번 버튼을 (명령어, 대상, 상세인자)로 변환
        p_idx = act_idx - 18
        target_type = 1 if p_idx % 2 == 0 else 2 # 1: 자신만, 2: 나 제외 전원
        p_idx //= 2

        if p_idx < 3: # Cmd 1: 열 교환
            return 1, target_type, [(0,1), (0,2), (1,2)][p_idx]
        p_idx -= 3
        if p_idx < 3: # Cmd 2: 행 교환
            return 2, target_type, [(0,1), (0,2), (1,2)][p_idx]
        p_idx -= 3
        if p_idx < 9: # Cmd 3: 성분 0 만들기
            return 3, target_type, (p_idx // 3, p_idx % 3)
        p_idx -= 9
        
        # Cmd 4: 두 성분 교환 (36쌍)
        pairs = [(i, j) for i in range(9) for j in range(i+1, 9)]
        return 4, target_type, pairs[p_idx]

    # --- 액션 마스킹 (상황에 따라 버튼 활성화/비활성화) ---
    def action_masks(self):
        mask = np.zeros(120, dtype=bool)
        if self.is_privilege_turn:
            # 특권 턴: 18~119번 버튼만 켬
            mask[18:120] = True
        else:
            # 일반 턴: 0~17번 버튼 중 이번 라운드에 안 쓴 칸만 켬
            for i in range(18):
                r, c, _ = self._decode_normal_action(i)
                if (r, c) not in self.used_cells_this_round[0]:
                    mask[i] = True
        return mask

    # --- 게임 내부 로직 ---
    def _calculate_x(self):
        for i in range(3):
            diag_sum = sum(p[i, i] for p in self.matrices)
            self.x_vector[i, 0] = math.floor(diag_sum / self.num_players)

    def _get_score(self, player_idx):
        mat = self.matrices[player_idx]
        if self.current_round <= 2:
            return np.sum(np.dot(mat, self.x_vector))
        else:
            return int(round(np.linalg.det(mat)))

    def _find_winner(self):
        scores = [(self._get_score(i), i) for i in range(self.num_players)]
        scores.sort(key=lambda x: x[0], reverse=True)
        unique_scores = []
        score_map = {}
        for s, idx in scores:
            if s not in score_map:
                unique_scores.append(s)
                score_map[s] = []
            score_map[s].append(idx)
            
        for i in range(min(3, len(unique_scores))):
            if len(score_map[unique_scores[i]]) == 1:
                return score_map[unique_scores[i]][0]
        return None

    def _apply_privilege(self, winner_idx, cmd, target_type, args):
        targets = [winner_idx] if target_type == 1 else [i for i in range(self.num_players) if i != winner_idx]
        for t in targets:
            if cmd == 1:
                c1, c2 = args
                self.matrices[t][:, [c1, c2]] = self.matrices[t][:, [c2, c1]]
            elif cmd == 2:
                r1, r2 = args
                self.matrices[t][[r1, r2], :] = self.matrices[t][[r2, r1], :]
            elif cmd == 3:
                r, c = args
                self.matrices[t][r, c] = 0
            elif cmd == 4:
                c1_idx, c2_idx = args
                r1, col1, r2, col2 = c1_idx//3, c1_idx%3, c2_idx//3, c2_idx%3
                self.matrices[t][r1, col1], self.matrices[t][r2, col2] = self.matrices[t][r2, col2], self.matrices[t][r1, col1]

    # --- 메인 스텝 함수 ---
    def step(self, action):
        reward = 0
        done = False

        # [A] AI가 특권을 행사하는 턴일 경우
        if self.is_privilege_turn:
            cmd, target_type, args = self._decode_privilege_action(action)
            self._apply_privilege(0, cmd, target_type, args)
            
            self.is_privilege_turn = False
            self.current_round += 1
            if self.current_round > 5:
                return self._end_game()
            return self._get_state(), reward, done, False, {}

        # [B] 일반 연산 턴일 경우
        # 1. AI 행동 적용
        row, col, value = self._decode_normal_action(action)
        self.matrices[0][row, col] += value
        self.used_cells_this_round[0].add((row, col))

        # 2. 봇 행동 적용
        for p in range(1, self.num_players):
            avail = [c for c in range(9) if (c//3, c%3) not in self.used_cells_this_round[p]]
            if avail:
                c = random.choice(avail)
                self.matrices[p][c//3, c%3] += random.choice([-1, 1])
                self.used_cells_this_round[p].add((c//3, c%3))

        self._calculate_x()
        self.actions_taken_in_round += 1

        # 라운드 종료 체크
        if self.actions_taken_in_round >= self.current_round:
            self.actions_taken_in_round = 0
            for p in range(self.num_players):
                self.used_cells_this_round[p].clear()

            winner_idx = self._find_winner()
            
            # AI가 우승했다면! 특권 턴 활성화
            if winner_idx == 0:
                self.is_privilege_turn = True
                return self._get_state(), 0, False, False, {} 
                
            # 봇이 우승했다면 무작위 특권 발동
            elif winner_idx is not None:
                cmd, t_type, args = self._decode_privilege_action(random.randint(18, 119))
                self._apply_privilege(winner_idx, cmd, t_type, args)
                
            self.current_round += 1

        if self.current_round > 5:
            return self._end_game()

        return self._get_state(), reward, done, False, {}

    def _end_game(self):
        results = []
        for i in range(self.num_players):
            det = int(round(np.linalg.det(self.matrices[i])))
            score = np.sum(np.dot(self.matrices[i], self.x_vector))
            results.append({'id': i, 'det': det, 'score': score, 'survive': det != 0})
        
        reward = -50 if not results[0]['survive'] else 10
        if results[0]['survive']:
            survivors = [r for r in results if r['survive']]
            survivors.sort(key=lambda x: (x['score'], x['det']), reverse=True)
            if survivors[0]['id'] == 0: reward = 100
            
        return self._get_state(), float(reward), True, False, {}

if __name__ == "__main__":
    print("👑 [3단계] 지령(특권) 사용법을 배우는 1세대 최종 AI 훈련 시작! (약 3~5분 소요)")
    env = MatrixGameEnvPhase3()
    model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    
    # 룰이 매우 복잡해졌으므로 충분한 경험을 쌓게 합니다.
    model.learn(total_timesteps=300000) 
    model.save("gen1_phase3_full_master")
    print("🎉 훈련 완료! 모든 룰을 마스터한 'gen1_phase3_full_master.zip' 뇌가 탄생했습니다!")