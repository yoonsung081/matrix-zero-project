import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO

class MatrixGameEnvGen2(gym.Env):
    def __init__(self):
        super(MatrixGameEnvGen2, self).__init__()
        self.action_space = spaces.Discrete(120)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(59,), dtype=np.float32)
        
        self.num_players = 6
        self.matrices = []
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        self.actions_taken_in_round = 0
        self.used_cells_this_round = [set() for _ in range(self.num_players)]
        self.is_privilege_turn = False
        
        # 🤖 [핵심!] 바보 봇 대신, 1세대 완전체 AI의 뇌를 적군으로 불러옵니다.
        # (경고가 뜰 수 있으나 예측용으로는 문제없이 로드됩니다)
        print("⏳ 과거 세대의 뇌(1세대 완전체)를 적군으로 복제 중...")
        self.bot_brain = MaskablePPO.load("gen1_phase3_full_master")

    # --- 상태 및 마스킹 헬퍼 함수 (각 플레이어 시점) ---
    def _get_state_for_player(self, player_idx, is_priv):
        state = []
        state.extend(self.matrices[player_idx].flatten())
        
        for i in range(self.num_players):
            if i != player_idx:
                enemy_mat = self.matrices[i].copy()
                np.fill_diagonal(enemy_mat, 0) # 남의 대각선은 0으로 가림
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
        """Stable-Baselines3가 0번 플레이어(학습 AI)의 마스킹을 확인할 때 호출하는 필수 함수"""
        return self._get_mask_for_player(0, self.is_privilege_turn) 
    # --- 디코딩 & 게임 로직 (기존 3단계와 동일) ---
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

    # --- 🤖 스텝 함수 (1세대 AI들의 지능적 대결!) ---
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

        # 1. 2세대 AI(학습 중인 모델) 행동 적용
        r, c, v = self._decode_normal_action(action)
        self.matrices[0][r, c] += v
        self.used_cells_this_round[0].add((r, c))

        # 2. 1세대 AI(적군 5명)들의 행동 예측 및 적용
        for p in range(1, self.num_players):
            obs_p = self._get_state_for_player(p, False)
            mask_p = self._get_mask_for_player(p, False)
            # 1세대 뇌를 사용해 최선의 수 도출
            bot_act, _ = self.bot_brain.predict(obs_p, action_masks=mask_p, deterministic=True)
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
                # 🚨 적(1세대 AI)이 우승했다면, 그들의 뇌로 치명적인 특권을 발동시킴!
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
    print("🔥 [2세대 자가 대국] 1세대 형님들과의 피 튀기는 데스매치가 시작됩니다! (약 5~10분 소요)")
    env = MatrixGameEnvGen2()
    
    # 🌟 전이 학습(Transfer Learning): 바닥부터 배우는 게 아니라 1세대의 뇌를 그대로 물려받아 시작!
    model = MaskablePPO.load("gen1_phase3_full_master", env=env)
    
    # AI들끼리의 대결이므로 진화하는 데 시간이 조금 더 걸립니다. 40만 판 진행!
    model.learn(total_timesteps=400000) 
    model.save("gen2_self_play_master")
    print("🎉 훈련 완료! 드디어 피바람 부는 리그에서 살아남은 2세대 AI(gen2_self_play_master)가 탄생했습니다!")