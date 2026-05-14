# matrix_game_env.py (개조된 환경 파일 예시)
import numpy as np
import math

class MatrixGameEnv:
    def __init__(self):
        self.num_players = 6
        self.reset()

    def reset(self):
        """매 게임이 끝날 때마다 초기화하는 함수"""
        self.matrices = [np.zeros((3, 3), dtype=int) for _ in range(self.num_players)]
        self.x_vector = np.zeros((3, 1), dtype=int)
        self.current_round = 1
        return self.get_state() # 초기 상태 반환

    def get_state(self, player_idx=0):
        """특정 플레이어(AI) 시점의 관측 상태를 반환 (불완전 정보 반영)"""
        # AI(자신)의 행렬은 온전히 보지만, 상대의 행렬은 대각 성분을 0이나 특정 마스킹 값으로 가려서 줌
        obs = []
        for i in range(self.num_players):
            mat_copy = self.matrices[i].copy()
            if i != player_idx: # 남의 행렬이면 대각선 블라인드 처리
                np.fill_diagonal(mat_copy, 0) # 임시로 0 처리 (AI가 구분할 수 있게)
            obs.append(mat_copy)
        return np.array(obs) # 6 x 3 x 3 텐서 형태

    def step(self, actions):
        """
        AI와 봇들이 선택한 행동(행, 열, 값)을 한 번에 받아서 환경을 업데이트
        actions: 각 플레이어의 행동 리스트. 예: [(0, 1, 1), (2, 2, -1), ...]
        """
        # 1. 행동 적용
        for p in range(self.num_players):
            row, col, val = actions[p]
            self.matrices[p][row, col] += val
        
        # 2. X 벡터 계산
        self.calculate_x() # 기존 코드의 로직 사용
        
        # 3. 우승자 특권 등 후속 처리 (특권 행동도 별도 처리 로직 필요)
        # winner = self.find_winner_with_privilege() ...
        
        # 4. 보상 산정 및 게임 종료 여부
        reward = self.get_reward(player_idx=0) # 학습할 1세대 AI(0번 조)의 보상
        done = (self.current_round >= 5)
        
        return self.get_state(player_idx=0), reward, done
    
    # ... 기존 calculate_x, get_score, find_winner 등의 함수는 그대로 유지 ...