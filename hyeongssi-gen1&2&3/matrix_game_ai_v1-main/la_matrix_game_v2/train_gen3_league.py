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

        # 2. 적군 행동 적용 (심화 리그: 1번 바보, 2~3번 1세대, 4~5번 2세대)
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