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

        # 2. 적군 행동 적용 (혼합 리그: 1~2번은 바보 봇, 3~5번은 1세대 챔피언)
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