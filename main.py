print("\n[System] Loading Evo-Matrix AI Engine v5.0 (Actor-Critic + Gen3 League)...")
import os, sys, gc, json, math, random
import numpy as np
from tqdm import tqdm

if 'VIRTUAL_ENV' in os.environ and 'CONDA_PREFIX' in os.environ:
    os.environ.pop('CONDA_PREFIX', None)

if sys.platform != 'win32':
    if 'CUDA_PATH' not in os.environ:
        for path in ['/usr/local/cuda', '/usr/local/cuda-13', '/usr/local/cuda-12']:
            if os.path.exists(path):
                os.environ['CUDA_PATH'] = path; break
else:
    if 'CUDA_PATH' not in os.environ:
        base = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
        if os.path.exists(base):
            vs = sorted([d for d in os.listdir(base) if d.startswith('v')], reverse=True)
            if vs:
                cd = os.path.join(base, vs[0])
                os.environ['CUDA_PATH'] = cd
                os.environ['PATH'] = os.path.join(cd,'bin') + os.pathsep + os.environ['PATH']

try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, AttributeError, RuntimeError):
    HAS_GPU = False

if not HAS_GPU:
    try:
        import numpy as cp
        print("Warning: CuPy 없음 — NumPy로 실행.")
    except ImportError:
        print("[Critical] pip install numpy"); sys.exit(1)

def to_cpu(a):
    return a.get() if HAS_GPU and hasattr(a, 'get') else a

PRIV_NAMES = ["행 교환", "열 교환", "원소 0화", "두 원소 교환"]

FEAT_DIM   = 30
HIDDEN_DIM = 128   # v4.0의 64 → 128 (표현력 증가)
ACT_DIM    = 18
PRIV_DIM   = 8

# ── Gen3 inference helper (numpy 전용, cupy 불필요) ──────────────
_gen3w = None

def load_gen3_weights(path="gen3_weights.json"):
    global _gen3w
    if _gen3w is not None:
        return True
    try:
        with open(path) as f:
            raw = json.load(f)
        _gen3w = {k: np.array(v, dtype=np.float32) for k, v in raw.items() if isinstance(v, list)}
        print(f"  Gen3 리그 가중치 로드 완료 ({path})")
        return True
    except Exception as e:
        print(f"  Gen3 가중치 로드 실패 ({e}) — Gen3 리그 상대 비활성화")
        return False

def gen3_act(matrices_np, t_idx, x_vec_np, rnd, is_priv, used_cells):
    """Gen3 행동 선택: numpy 행렬 입력, action_idx 반환"""
    if _gen3w is None:
        return None
    # Obs 구성 (59차원: 자신 9 + 상대 5×9(대각0) + x3 + round1 + priv1)
    obs = list(matrices_np[t_idx].flatten())
    for p in range(6):
        if p == t_idx: continue
        m = matrices_np[p].copy().astype(float)
        m[0,0] = m[1,1] = m[2,2] = 0.0
        obs.extend(m.flatten())
    obs.extend(x_vec_np.flatten())
    obs.append(float(rnd))
    obs.append(1.0 if is_priv else 0.0)
    obs = np.array(obs, dtype=np.float32)

    h1 = np.tanh(_gen3w['pn_w1'] @ obs  + _gen3w['pn_b1'])
    h2 = np.tanh(_gen3w['pn_w2'] @ h1   + _gen3w['pn_b2'])
    logits = _gen3w['act_w'] @ h2 + _gen3w['act_b']

    if is_priv:
        logits[:18] = -1e9
    else:
        logits[18:] = -1e9  # privilege 범위 차단
        for r, c in used_cells:
            logits[(r*3+c)*2]   = -1e9
            logits[(r*3+c)*2+1] = -1e9
    return int(np.argmax(logits))


class MatrixGame:
    def __init__(self):
        self.num_players   = 6
        self.matrices      = [cp.zeros((3,3), dtype=int) for _ in range(self.num_players)]
        self.x_vector      = cp.zeros((3,1),  dtype=int)
        self.current_round = 1

        # ── He 초기화 (v4.0과 동일, 단 HIDDEN_DIM=128) ─────────
        def he(fan_in, out):
            return cp.random.randn(fan_in, out) * np.sqrt(2.0 / fan_in)

        # 일반 행동 네트워크: 공유 레이어 + 정책 헤드 + 가치 헤드
        self.w1   = he(FEAT_DIM,   HIDDEN_DIM)
        self.b1   = cp.zeros(HIDDEN_DIM)
        self.w_pi = he(HIDDEN_DIM, ACT_DIM)     # 정책 헤드
        self.b_pi = cp.zeros(ACT_DIM)
        self.w_v  = he(HIDDEN_DIM, 1)           # 가치 헤드 (NEW)
        self.b_v  = cp.zeros(1)

        # 특권 네트워크: 동일 구조
        self.pw1   = he(FEAT_DIM,   HIDDEN_DIM)
        self.pb1   = cp.zeros(HIDDEN_DIM)
        self.pw_pi = he(HIDDEN_DIM, PRIV_DIM)
        self.pb_pi = cp.zeros(PRIV_DIM)
        self.pw_v  = he(HIDDEN_DIM, 1)
        self.pb_v  = cp.zeros(1)

        # Adam 옵티마이저 상태
        self.lr      = 3e-4
        self.beta1   = 0.9
        self.beta2   = 0.999
        self.eps_a   = 1e-8
        self.adam_t  = 0

        all_params = [self.w1, self.b1, self.w_pi, self.b_pi, self.w_v, self.b_v,
                      self.pw1, self.pb1, self.pw_pi, self.pb_pi, self.pw_v, self.pb_v]
        self.m = [cp.zeros_like(p) for p in all_params]
        self.v = [cp.zeros_like(p) for p in all_params]

        # 하이퍼파라미터
        self.gamma        = 0.95
        self.entropy_coef = 0.02   # 탐험 강화 (v4.0 0.01 → 0.02)
        self.value_coef   = 0.5    # critic loss 가중치
        self.max_grad_norm = 1.0
        self.reward_baseline = 0.0
        self.baseline_alpha  = 0.05

        self.used_cells    = [set() for _ in range(self.num_players)]
        self.learner_history = []

    # ── 게임 로직 ─────────────────────────────────────────────────

    def calculate_x(self):
        stacked = cp.stack(self.matrices)
        diags   = cp.diagonal(stacked, axis1=1, axis2=2)
        self.x_vector = cp.floor(cp.sum(diags, axis=0) / self.num_players).astype(int).reshape(3, 1)

    def get_score(self, p, mode="auto"):
        m = self.matrices[p].astype(cp.float32)
        if mode == "ax" or (mode == "auto" and self.current_round <= 2):
            return float(to_cpu(cp.sum(cp.dot(m, self.x_vector.astype(cp.float32)))))
        return float(to_cpu(cp.round(cp.linalg.det(m))))

    def get_round_winner(self):
        scores = [self.get_score(i) for i in range(self.num_players)]
        sm = {}
        for i, s in enumerate(scores):
            sm.setdefault(s, []).append(i)
        for s in sorted(sm.keys(), reverse=True)[:3]:
            if len(sm[s]) == 1: return sm[s][0]
        return None

    def get_final_winner(self):
        surv = [(self.get_score(i,'ax'), self.get_score(i,'det'), i)
                for i in range(self.num_players) if self.get_score(i,'det') != 0]
        if not surv: return []
        best_ax = max(s[0] for s in surv)
        top = [s for s in surv if s[0] == best_ax]
        best_det = max(s[1] for s in top)
        return [s[2] for s in top if s[1] == best_det]

    def get_reward(self, p):
        # 스케일 조정된 보상 (v4.0과 동일)
        if self.current_round == 5:
            det = self.get_score(p, 'det')
            if det == 0: return -3.0
            return 5.0 if p in self.get_final_winner() else 0.5
        return 1.0 if self.get_round_winner() == p else 0.0

    def reset_game(self):
        self.matrices      = [cp.zeros((3,3), dtype=int) for _ in range(self.num_players)]
        self.x_vector      = cp.zeros((3,1), dtype=int)
        self.current_round = 1
        for s in self.used_cells: s.clear()

    # ── 특징 벡터 ─────────────────────────────────────────────────

    def _feat(self, p):
        f = np.zeros(FEAT_DIM, dtype=np.float32)
        f[0:9]  = to_cpu(self.matrices[p].flatten()).astype(np.float32)
        f[9:12] = to_cpu(self.x_vector.flatten()).astype(np.float32)
        f[12]   = float(self.current_round)
        sc = np.array([self.get_score(i) for i in range(self.num_players)], np.float32)
        f[13:19] = sc / (np.max(np.abs(sc)) + 1e-8)
        dt = np.array([float(to_cpu(cp.round(cp.linalg.det(
            self.matrices[i].astype(cp.float32))))) for i in range(self.num_players)], np.float32)
        f[19:25] = dt / (np.max(np.abs(dt)) + 1e-8)
        f[25] = sum(1 for s in sc if s > sc[p]) / max(self.num_players-1, 1)
        f[26]  = 1.0
        return cp.array(f)

    # ── Actor-Critic 순전파 ───────────────────────────────────────

    def _ac_forward(self, feat, w1, b1, w_pi, b_pi, w_v, b_v):
        h      = cp.maximum(0, cp.dot(feat, w1) + b1)          # 공유 은닉층
        logits = cp.dot(h, w_pi) + b_pi                         # 정책 logits
        value  = float(to_cpu(cp.dot(h, w_v) + b_v)[0])        # 상태 가치
        return logits, h, value

    def _softmax(self, logits):
        e = cp.exp(logits - cp.max(logits))
        return e / (cp.sum(e) + 1e-8)

    def _sample(self, logits, temp):
        l = (logits - cp.max(logits)) / max(temp, 1e-6)
        p = cp.exp(l); p /= cp.sum(p)
        p_np = to_cpu(p).astype(np.float64); p_np /= p_np.sum()
        return int(np.random.choice(len(p_np), p=p_np))

    def _mask(self, p_idx):
        mask = cp.ones(ACT_DIM, dtype=cp.float32)
        for r, c in self.used_cells[p_idx]:
            mask[(r*3+c)*2] = mask[(r*3+c)*2+1] = 0.0
        return mask

    def _current_ac(self):
        return (self.w1, self.b1, self.w_pi, self.b_pi, self.w_v, self.b_v)

    # ── 에이전트 ──────────────────────────────────────────────────

    def agent_random(self, p):
        avail = [(r,c) for r in range(3) for c in range(3) if (r,c) not in self.used_cells[p]]
        if not avail: avail = [(r,c) for r in range(3) for c in range(3)]
        r, c = random.choice(avail)
        self.matrices[p][r,c] += random.choice([-1,1])
        self.used_cells[p].add((r,c))

    def agent_learner_act(self, p, ac, temp=0.0):
        w1,b1,wp,bp,wv,bv = ac
        feat = self._feat(p)
        logits, h, value = self._ac_forward(feat, w1,b1,wp,bp,wv,bv)
        masked = logits + (1.0 - self._mask(p)) * cp.float32(-1e9)
        a = self._sample(masked, temp) if temp > 0 else int(to_cpu(cp.argmax(masked)))
        ri, ci = (a//2)//3, (a//2)%3
        self.matrices[p][ri,ci] += 1 if a%2==0 else -1
        self.used_cells[p].add((ri,ci))
        return feat, h, a, logits, value

    def agent_gen3(self, p):
        """Gen3 행동 — gen3_weights.json 사용"""
        mats_np = [to_cpu(m) for m in self.matrices]
        x_np    = to_cpu(self.x_vector)
        a = gen3_act(mats_np, p, x_np, self.current_round, False, self.used_cells[p])
        if a is None:
            self.agent_random(p); return
        ri, ci = (a//2)//3, (a//2)%3
        v = 1 if a%2==0 else -1
        self.matrices[p][ri,ci] += v
        self.used_cells[p].add((ri,ci))

    def agent_gen3_priv(self, p):
        """Gen3 특권 선택"""
        mats_np = [to_cpu(m) for m in self.matrices]
        x_np    = to_cpu(self.x_vector)
        a = gen3_act(mats_np, p, x_np, self.current_round, True, set())
        if a is None or a < 18:
            self.agent_random_privilege(p); return
        # Gen3 특권 디코딩 및 적용
        pi = a - 18
        scope = 'self' if pi % 2 == 0 else 'others'
        pi //= 2
        pairs3 = [(0,1),(0,2),(1,2)]
        if pi < 3:   cmd, args = 'swap_col', pairs3[pi]
        elif pi < 6: cmd, args = 'swap_row', pairs3[pi-3]
        elif pi < 15: cmd, args = 'zero', (((pi-6)//3), ((pi-6)%3))
        else:
            all_pairs = [(i,j) for i in range(9) for j in range(i+1,9)]
            cmd, args = 'swap_elem', all_pairs[pi-15]
        targets = [p] if scope=='self' else [t for t in range(self.num_players) if t!=p]
        for t in targets:
            m = self.matrices[t]
            if cmd=='swap_col':
                c1,c2=args; tmp=m[:,c1].copy(); m[:,c1]=m[:,c2]; m[:,c2]=tmp
            elif cmd=='swap_row':
                r1,r2=args; tmp=m[r1].copy(); m[r1]=m[r2]; m[r2]=tmp
            elif cmd=='zero':
                m[args[0],args[1]]=0
            elif cmd=='swap_elem':
                i1,i2=args; r1,c1=i1//3,i1%3; r2,c2=i2//3,i2%3
                m[r1,c1],m[r2,c2] = m[r2,c2],m[r1,c1]

    # ── 특권 ─────────────────────────────────────────────────────

    def agent_learner_privilege(self, p, ac, temp=0.0):
        w1,b1,wp,bp,wv,bv = ac
        feat = self._feat(p)
        logits, h, value = self._ac_forward(feat, w1,b1,wp,bp,wv,bv)
        a = self._sample(logits, temp) if temp > 0 else int(to_cpu(cp.argmax(logits)))
        self.apply_privilege(p, a//2, a%2)
        return feat, h, a, logits, value

    def _priv_ac(self):
        return (self.pw1, self.pb1, self.pw_pi, self.pb_pi, self.pw_v, self.pb_v)

    def apply_privilege(self, winner, priv_type, scope):
        effect = (self._best_priv(priv_type, winner, minimize=False) if scope==0
                  else self._best_priv_global(priv_type, winner))
        if scope == 0:
            self._do_priv(priv_type, winner, effect)
        else:
            for t in range(self.num_players):
                if t != winner: self._do_priv(priv_type, t, effect)

    def agent_random_privilege(self, winner):
        self.apply_privilege(winner, random.randint(0,3), random.randint(0,1))

    def _eval(self, m):
        mf = m.astype(cp.float32)
        if self.current_round <= 2:
            return float(to_cpu(cp.sum(cp.dot(mf, self.x_vector.astype(cp.float32)))))
        return float(to_cpu(cp.round(cp.linalg.det(mf))))

    def _best_priv(self, pt, ti, minimize=False):
        m = self.matrices[ti]
        best, bs = None, float('inf') if minimize else -float('inf')
        better = (lambda s: s < bs) if minimize else (lambda s: s > bs)
        if pt == 0:
            for i,j in [(0,1),(0,2),(1,2)]:
                n=m.copy(); tmp=n[i].copy(); n[i]=n[j]; n[j]=tmp; s=self._eval(n)
                if better(s): bs=s; best=(i,j)
        elif pt == 1:
            for i,j in [(0,1),(0,2),(1,2)]:
                n=m.copy(); tmp=n[:,i].copy(); n[:,i]=n[:,j]; n[:,j]=tmp; s=self._eval(n)
                if better(s): bs=s; best=(i,j)
        elif pt == 2:
            for r in range(3):
                for c in range(3):
                    if int(to_cpu(m[r,c]))==0: continue
                    n=m.copy(); n[r,c]=0; s=self._eval(n)
                    if better(s): bs=s; best=(r,c)
        elif pt == 3:
            pos=[(r,c) for r in range(3) for c in range(3)]
            for k in range(len(pos)):
                for l in range(k+1,len(pos)):
                    r1,c1=pos[k]; r2,c2=pos[l]
                    if int(to_cpu(m[r1,c1]))==int(to_cpu(m[r2,c2])): continue
                    n=m.copy(); tmp=n[r1,c1].copy(); n[r1,c1]=n[r2,c2]; n[r2,c2]=tmp; s=self._eval(n)
                    if better(s): bs=s; best=(r1,c1,r2,c2)
        return best

    def _best_priv_global(self, pt, wi):
        others = [t for t in range(self.num_players) if t!=wi]
        best, bt = None, float('inf')
        def try_effect(effect):
            nonlocal best, bt
            total = 0
            for t in others:
                n = self.matrices[t].copy()
                self._do_priv_on(pt, n, effect)
                total += self._eval(n)
            if total < bt: bt=total; best=effect
        if pt in [0,1]:
            for pair in [(0,1),(0,2),(1,2)]: try_effect(pair)
        elif pt == 2:
            for r in range(3):
                for c in range(3): try_effect((r,c))
        elif pt == 3:
            pos=[(r,c) for r in range(3) for c in range(3)]
            for k in range(len(pos)):
                for l in range(k+1,len(pos)): try_effect((pos[k][0],pos[k][1],pos[l][0],pos[l][1]))
        return best

    def _do_priv(self, pt, ti, effect):
        if effect is None: return
        self._do_priv_on(pt, self.matrices[ti], effect)

    def _do_priv_on(self, pt, m, effect):
        if effect is None: return
        if pt==0:   i,j=effect; tmp=m[i].copy(); m[i]=m[j]; m[j]=tmp
        elif pt==1: i,j=effect; tmp=m[:,i].copy(); m[:,i]=m[:,j]; m[:,j]=tmp
        elif pt==2: r,c=effect; m[r,c]=0
        elif pt==3: r1,c1,r2,c2=effect; tmp=m[r1,c1].copy(); m[r1,c1]=m[r2,c2]; m[r2,c2]=tmp

    # ── Adam ─────────────────────────────────────────────────────

    def _adam(self, grads, param_slice):
        """param_slice: self.m/v의 인덱스 슬라이스, 해당 파라미터 리스트 반환"""
        self.adam_t += 1
        t = self.adam_t
        bc1, bc2 = 1.0 - self.beta1**t, 1.0 - self.beta2**t
        updated = []
        for i, g in enumerate(grads):
            idx = param_slice.start + i
            self.m[idx] = self.beta1*self.m[idx] + (1-self.beta1)*g
            self.v[idx] = self.beta2*self.v[idx] + (1-self.beta2)*g*g
            updated.append(self.lr * (self.m[idx]/bc1) / (cp.sqrt(self.v[idx]/bc2) + self.eps_a))
        return updated

    def _clip_grads(self, grads):
        norm = cp.sqrt(sum(cp.sum(g**2) for g in grads))
        if norm > self.max_grad_norm:
            c = self.max_grad_norm / (norm + 1e-6)
            grads = [g*c for g in grads]
        return grads

    # ── Actor-Critic 업데이트 ─────────────────────────────────────

    def update_weights(self, episode_actions, episode_privs, round_rewards):
        """
        episode_actions: [(feat, h, a_idx, logits, value, r_num), ...]
        round_rewards:   [r1_reward, r2_reward, ..., r5_reward]
        """
        gamma = self.gamma

        # ── 일반 행동 파라미터 목록 (Adam 인덱스 0~5) ───────────
        #  [w1, b1, w_pi, b_pi, w_v, b_v]

        for feat, h, a_idx, logits, value, r_num in episode_actions:
            # 스텝 t의 리턴: 남은 라운드 보상 합산
            G = sum(round_rewards[k] * (gamma ** (k - (r_num-1)))
                    for k in range(r_num-1, 5))

            advantage = G - value   # Actor-Critic advantage (TD error)
            advantage = float(np.clip(advantage, -3.0, 3.0))  # 폭발 방지
            # baseline 업데이트 (EMA)
            self.reward_baseline += self.baseline_alpha * (G - self.reward_baseline)

            probs = self._softmax(logits)
            e_a   = cp.zeros(ACT_DIM, dtype=cp.float32); e_a[a_idx] = 1.0

            # Actor gradient: advantage * (e_a - probs) + entropy
            policy_grad  = advantage * (e_a - probs)
            entropy_grad = self.entropy_coef * (-(cp.log(probs + 1e-8) + 1.0))
            grad_pi      = policy_grad + entropy_grad

            # Critic gradient: minimize 0.5*(G - V)^2  →  dL/dV = -(G-V)
            grad_v = cp.array([-(advantage) * self.value_coef], dtype=cp.float32)

            # Backprop through shared layer
            d_h_pi = cp.dot(self.w_pi, grad_pi)   * (h > 0)
            d_h_v  = cp.dot(self.w_v,  grad_v)    * (h > 0)
            d_h    = d_h_pi + d_h_v

            grads = self._clip_grads([
                cp.outer(feat, d_h),     # dw1
                d_h,                     # db1
                cp.outer(h, grad_pi),    # dw_pi
                grad_pi,                 # db_pi
                cp.outer(h, grad_v),     # dw_v
                grad_v,                  # db_v
            ])
            deltas = self._adam(grads, slice(0, 6))
            (self.w1, self.b1, self.w_pi, self.b_pi,
             self.w_v, self.b_v) = [p + d for p, d in zip(
                [self.w1, self.b1, self.w_pi, self.b_pi, self.w_v, self.b_v], deltas)]

        # ── 특권 파라미터 목록 (Adam 인덱스 6~11) ────────────────

        for feat, h, a_idx, logits, value, r_num in episode_privs:
            G = sum(round_rewards[k] * (gamma ** (k - (r_num-1)))
                    for k in range(r_num-1, 5))
            advantage = float(np.clip(G - value, -3.0, 3.0))

            probs = self._softmax(logits)
            e_a   = cp.zeros(PRIV_DIM, dtype=cp.float32); e_a[a_idx] = 1.0
            policy_grad  = advantage * (e_a - probs)
            entropy_grad = self.entropy_coef * (-(cp.log(probs + 1e-8) + 1.0))
            grad_pi      = policy_grad + entropy_grad
            grad_v       = cp.array([-(advantage) * self.value_coef], dtype=cp.float32)

            d_h = (cp.dot(self.pw_pi, grad_pi) + cp.dot(self.pw_v, grad_v)) * (h > 0)
            grads = self._clip_grads([
                cp.outer(feat, d_h), d_h,
                cp.outer(h, grad_pi), grad_pi,
                cp.outer(h, grad_v), grad_v,
            ])
            deltas = self._adam(grads, slice(6, 12))
            (self.pw1, self.pb1, self.pw_pi, self.pb_pi,
             self.pw_v, self.pb_v) = [p + d for p, d in zip(
                [self.pw1, self.pb1, self.pw_pi, self.pb_pi, self.pw_v, self.pb_v], deltas)]

    # ── 스냅샷 ────────────────────────────────────────────────────

    def _snapshot(self):
        return {k: getattr(self, k).copy() for k in
                ['w1','b1','w_pi','b_pi','w_v','b_v']}

    # ── 시뮬레이션 ────────────────────────────────────────────────

    def run_simulation(self, num_episodes=1_200_000):
        has_gen3 = load_gen3_weights()
        print(f"\n학습 시작: {num_episodes:,}회  Gen3 리그: {'ON' if has_gen3 else 'OFF'}")
        print("구성: Random 20% / 리그(스냅샷) 25% / OldSelf 45% / Gen3 10%\n")

        results_data = []
        self.learner_history.append(self._snapshot())

        TEMP_START, TEMP_END = 1.0, 0.05
        win_cnt = surv_cnt = 0

        try:
         for ep in tqdm(range(num_episodes)):
            self.reset_game()
            episode_actions = []
            episode_privs   = []
            round_rewards   = [0.0] * 5   # 라운드별 보상

            temp = TEMP_START - (TEMP_START - TEMP_END) * (ep / num_episodes)

            if ep % 2000 == 0:
                self.learner_history.append(self._snapshot())

            # 리그 상대 구성
            opponents = []
            for _ in range(5):
                rv = random.random()
                if rv < 0.20:                        opponents.append("Random")
                elif rv < 0.45 and len(self.learner_history) > 1:
                    opponents.append("League")
                elif rv < 0.90:                      opponents.append("OldSelf")
                else:                                opponents.append("Gen3")

            ac       = self._current_ac()
            priv_ac  = self._priv_ac()

            for r in range(1, 6):
                self.current_round = r
                for s in self.used_cells: s.clear()

                for p in range(self.num_players):
                    for _ in range(r):
                        if p == 0:
                            feat, h, a, logits, val = self.agent_learner_act(ac=ac, p=0, temp=temp)
                            episode_actions.append((feat, h, a, logits, val, r))
                        else:
                            opp = opponents[p-1]
                            if opp == "Random":
                                self.agent_random(p)
                            elif opp == "Gen3" and has_gen3:
                                self.agent_gen3(p)
                            elif opp == "League":
                                snap = random.choice(self.learner_history)
                                self.agent_learner_act(
                                    p=p, ac=(snap['w1'],snap['b1'],snap['w_pi'],snap['b_pi'],snap['w_v'],snap['b_v'])
                                )
                            else:  # OldSelf
                                snap = self.learner_history[-1]
                                self.agent_learner_act(
                                    p=p, ac=(snap['w1'],snap['b1'],snap['w_pi'],snap['b_pi'],snap['w_v'],snap['b_v'])
                                )

                self.calculate_x()

                rw = self.get_round_winner()
                if rw == 0:
                    round_rewards[r-1] += 1.0
                    feat, h, a, logits, val = self.agent_learner_privilege(
                        p=0, ac=priv_ac, temp=temp)
                    episode_privs.append((feat, h, a, logits, val, r))
                elif rw is not None:
                    if has_gen3 and opponents[rw-1] == "Gen3":
                        self.agent_gen3_priv(rw)
                    else:
                        self.agent_random_privilege(rw)

            # 최종 보상을 마지막 라운드에 합산
            final_r = self.get_reward(0)
            round_rewards[4] += final_r

            self.update_weights(episode_actions, episode_privs, round_rewards)

            total_r = sum(round_rewards)
            if total_r > 0: surv_cnt += 1
            if final_r >= 5.0: win_cnt += 1

            if ep > 0 and ep % 10000 == 0:
                tqdm.write(f"  ep={ep:,}: 승률={win_cnt/ep*100:.1f}%  "
                           f"생존율={surv_cnt/ep*100:.1f}%")

            if HAS_GPU and ep % 500 == 0:
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

            if ep > num_episodes - 2000:
                results_data.append({
                    "matrix": to_cpu(self.matrices[0]).tolist(),
                    "x":      to_cpu(self.x_vector).flatten().tolist(),
                    "score":  self.get_score(0)
                })

        except KeyboardInterrupt:
            print("\n[중단됨] 가중치 저장 중...")
            self._save()
            print("저장 완료. 종료.")
            return results_data

        print(f"\n최종: 승률={win_cnt/num_episodes*100:.2f}%  "
              f"생존율={surv_cnt/num_episodes*100:.2f}%")
        self._save()
        return results_data

    def _save(self, path="strategy_weights.json"):
        data = {}
        for k in ['w1','b1','w_pi','b_pi','w_v','b_v',
                  'pw1','pb1','pw_pi','pb_pi','pw_v','pb_v']:
            data[k] = to_cpu(getattr(self, k)).tolist()
        # battle.html 호환성을 위해 w2/b2 별칭도 저장
        data['w2']  = data['w_pi'];  data['b2']  = data['b_pi']
        data['pw2'] = data['pw_pi']; data['pb2'] = data['pb_pi']
        data['feat_dim'] = FEAT_DIM; data['hidden_dim'] = HIDDEN_DIM
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"가중치 저장 완료: {path}")

    def load_weights(self, path="strategy_weights.json"):
        if not os.path.exists(path): return False
        try:
            with open(path) as f: data = json.load(f)
            key_map = {'w1':'w1','b1':'b1','w_pi':['w_pi','w2'],'b_pi':['b_pi','b2'],
                       'w_v':'w_v','b_v':'b_v',
                       'pw1':'pw1','pb1':'pb1','pw_pi':['pw_pi','pw2'],'pb_pi':['pb_pi','pb2'],
                       'pw_v':'pw_v','pb_v':'pb_v'}
            for attr, keys in key_map.items():
                keys = [keys] if isinstance(keys, str) else keys
                for k in keys:
                    if k in data:
                        setattr(self, attr, cp.array(data[k])); break
            # Adam 리셋
            all_p = [self.w1,self.b1,self.w_pi,self.b_pi,self.w_v,self.b_v,
                     self.pw1,self.pb1,self.pw_pi,self.pb_pi,self.pw_v,self.pb_v]
            self.m = [cp.zeros_like(p) for p in all_p]
            self.v = [cp.zeros_like(p) for p in all_p]
            self.adam_t = 0
            print("가중치 로드 완료 (v5.0 또는 v4.0 호환)")
            return True
        except Exception as e:
            print(f"로드 실패: {e}"); return False

    # ── 대화형 대전 ───────────────────────────────────────────────

    def play_interactive(self):
        self.reset_game()
        print("\n" + "="*50 + "\n [MATCH] AI (T1) vs YOU (T2) vs Bots\n" + "="*50)
        for r in range(1, 6):
            self.current_round = r
            for s in self.used_cells: s.clear()
            print(f"\n{'='*20} ROUND {r} {'='*20}")
            for p in range(self.num_players):
                if p == 0:
                    for _ in range(r):
                        self.agent_learner_act(p=p, ac=self._current_ac())
                    print("[AI] 행동 완료.")
                elif p == 1:
                    print(f"\n[YOUR TURN]\n{to_cpu(self.matrices[p])}")
                    for a in range(r):
                        while True:
                            try:
                                mv = input(f"행동 {a+1}/{r} (행 열 값): ").split()
                                row,col,val = int(mv[0]),int(mv[1]),int(mv[2])
                                if val not in [-1,1] or not(0<=row<=2 and 0<=col<=2): raise ValueError
                                if (row,col) in self.used_cells[p]: print("이미 사용한 칸"); continue
                                self.matrices[p][row,col]+=val; self.used_cells[p].add((row,col)); break
                            except Exception: print("형식: 행(0-2) 열(0-2) 값(-1/1)")
                else:
                    for _ in range(r): self.agent_random(p)
            self.calculate_x()
            scores = [self.get_score(i) for i in range(self.num_players)]
            print(f"X={to_cpu(self.x_vector).flatten()}  점수={[round(s,1) for s in scores]}")
            w = self.get_round_winner()
            if w is not None:
                print(f"라운드 승자: T{w+1}")
                if w == 0:
                    _, _, a, _, _ = self.agent_learner_privilege(p=0, ac=self._priv_ac())
                    print(f"[AI 특권] {PRIV_NAMES[a//2]} / {'내팀' if a%2==0 else '전타팀'}")
                elif w != 1: self.agent_random_privilege(w)
        print("\n[최종]")
        for i in range(self.num_players):
            d=self.get_score(i,'det'); ax=self.get_score(i,'ax')
            lbl="AI" if i==0 else "YOU" if i==1 else f"Bot{i+1}"
            st="WINNER" if i in self.get_final_winner() else "SURVIVED" if d!=0 else "ELIM"
            print(f" {lbl}: Ax={ax:.1f} det={d} [{st}]")


if __name__ == "__main__":
    game = MatrixGame()
    print("=== Evo-Matrix AI Engine v5.0 ===")
    print("Actor-Critic | Gen3 리그 상대 | HIDDEN=128 | Adam | 1.2M 에피소드")
    print("1. 대전 테스트")
    print("2. 신규 학습 (1,200,000 에피소드)")
    print("3. 이어서 학습 (기존 가중치 로드)")

    choice = input("\n선택 (1/2/3): ").strip()
    if choice == "1":
        if game.load_weights(): game.play_interactive()
        else: print("[오류] 학습된 가중치 없음.")
    elif choice == "2":
        game.run_simulation(1_200_000)
    elif choice == "3":
        game.load_weights()
        game.run_simulation(1_200_000)
    else:
        print("잘못된 선택.")
