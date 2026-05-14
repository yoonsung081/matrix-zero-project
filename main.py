print("\n[System] Evo-Matrix AI Engine v6.0 (PPO + GAE)")
import os, json, random
import numpy as np
from tqdm import tqdm

# ── Gen3 weights ──────────────────────────────────────────────────
_gen3w = None

def load_gen3_weights(path="gen3_weights.json"):
    global _gen3w
    if _gen3w is not None: return True
    try:
        with open(path, encoding='utf-8') as f: raw = json.load(f)
        _gen3w = {k: np.array(v, np.float32) for k, v in raw.items() if isinstance(v, list)}
        print(f"  Gen3 가중치 로드 완료 ({path})")
        return True
    except Exception as e:
        print(f"  Gen3 가중치 없음 ({e})"); return False

def gen3_act(matrices, t_idx, x_vec, rnd, is_priv, used_cells):
    if _gen3w is None: return None
    obs = list(matrices[t_idx].flatten().astype(float))
    for p in range(6):
        if p == t_idx: continue
        m = matrices[p].copy().astype(float)
        m[0,0] = m[1,1] = m[2,2] = 0.0
        obs.extend(m.flatten())
    obs.extend(x_vec.flatten())
    obs.append(float(rnd)); obs.append(1.0 if is_priv else 0.0)
    obs = np.array(obs, np.float32)
    h1 = np.tanh(_gen3w['pn_w1'] @ obs + _gen3w['pn_b1'])
    h2 = np.tanh(_gen3w['pn_w2'] @ h1  + _gen3w['pn_b2'])
    logits = _gen3w['act_w'] @ h2 + _gen3w['act_b']
    if is_priv:
        logits[:18] = -1e9
    else:
        logits[18:] = -1e9
        for r, c in used_cells:
            logits[(r*3+c)*2] = logits[(r*3+c)*2+1] = -1e9
    return int(np.argmax(logits))

# ── Hyperparameters ───────────────────────────────────────────────
FEAT_DIM       = 30
HIDDEN_DIM     = 128
ACT_DIM        = 18
PRIV_DIM       = 8
ROLLOUT_NORMAL = 4096
ROLLOUT_PRIV   = 512
PPO_EPOCHS     = 4
MINI_BATCH     = 256
CLIP_EPS       = 0.2
GAMMA          = 0.95
GAE_LAM        = 0.95
VALUE_COEF     = 0.5
ENTROPY_COEF   = 0.01
LR             = 3e-4
MAX_GRAD_NORM  = 0.5
PRIV_NAMES     = ["행 교환", "열 교환", "원소 0화", "두 원소 교환"]

# ── Rollout Buffer ────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.feats=[]; self.actions=[]; self.log_probs=[]
        self.values=[]; self.rewards=[]; self.dones=[]; self.masks=[]

    def add(self, feat, action, log_prob, value, reward, done, mask):
        self.feats.append(feat.copy())
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.masks.append(mask.copy())

    def __len__(self): return len(self.feats)
    def ready(self):   return len(self) >= self.max_size

    def compute_gae(self):
        n    = len(self.feats)
        vals = np.array(self.values,  np.float32)
        rews = np.array(self.rewards, np.float32)
        dns  = np.array(self.dones,   np.float32)
        adv  = np.zeros(n, np.float32)
        gae  = 0.0
        for t in reversed(range(n)):
            nv    = vals[t+1] if t+1 < n else 0.0
            mask  = 1.0 - dns[t]
            delta = rews[t] + GAMMA * nv * mask - vals[t]
            gae   = delta + GAMMA * GAE_LAM * mask * gae
            adv[t] = gae
        return adv, adv + vals

# ── MatrixGame (game logic + PPO agent) ───────────────────────────
class MatrixGame:
    def __init__(self):
        self.num_players  = 6
        self.matrices     = [np.zeros((3,3), int) for _ in range(6)]
        self.x_vector     = np.zeros((3,1), int)
        self.current_round = 1
        self.used_cells   = [set() for _ in range(6)]
        self.learner_history = []

        def he(fi, fo): return np.random.randn(fi, fo).astype(np.float32) * np.sqrt(2./fi)

        self.w1   = he(FEAT_DIM, HIDDEN_DIM)
        self.b1   = np.zeros(HIDDEN_DIM, np.float32)
        self.w_pi = he(HIDDEN_DIM, ACT_DIM)
        self.b_pi = np.zeros(ACT_DIM, np.float32)
        self.w_v  = he(HIDDEN_DIM, 1)
        self.b_v  = np.zeros(1, np.float32)

        self.pw1   = he(FEAT_DIM, HIDDEN_DIM)
        self.pb1   = np.zeros(HIDDEN_DIM, np.float32)
        self.pw_pi = he(HIDDEN_DIM, PRIV_DIM)
        self.pb_pi = np.zeros(PRIV_DIM, np.float32)
        self.pw_v  = he(HIDDEN_DIM, 1)
        self.pb_v  = np.zeros(1, np.float32)

        all_p = [self.w1,self.b1,self.w_pi,self.b_pi,self.w_v,self.b_v,
                 self.pw1,self.pb1,self.pw_pi,self.pb_pi,self.pw_v,self.pb_v]
        self.m = [np.zeros_like(p) for p in all_p]
        self.v = [np.zeros_like(p) for p in all_p]
        self.adam_t = 0

    # ── Game logic ────────────────────────────────────────────────
    def reset_game(self):
        self.matrices      = [np.zeros((3,3), int) for _ in range(6)]
        self.x_vector      = np.zeros((3,1), int)
        self.current_round = 1
        for s in self.used_cells: s.clear()

    def calculate_x(self):
        diags = np.array([np.diag(m) for m in self.matrices])
        self.x_vector = np.floor(diags.sum(0) / 6).astype(int).reshape(3,1)

    def get_score(self, p, mode="auto"):
        m = self.matrices[p].astype(float)
        if mode=="ax" or (mode=="auto" and self.current_round<=2):
            return float(np.sum(m @ self.x_vector.astype(float)))
        return float(round(np.linalg.det(m)))

    def get_round_winner(self):
        scores = [self.get_score(i) for i in range(6)]
        sm = {}
        for i,s in enumerate(scores): sm.setdefault(s,[]).append(i)
        for s in sorted(sm.keys(), reverse=True)[:3]:
            if len(sm[s])==1: return sm[s][0]
        return None

    def get_final_winner(self):
        surv = [(self.get_score(i,'ax'), self.get_score(i,'det'), i)
                for i in range(6) if self.get_score(i,'det')!=0]
        if not surv: return []
        ba = max(s[0] for s in surv)
        top = [s for s in surv if s[0]==ba]
        bd = max(s[1] for s in top)
        return [s[2] for s in top if s[1]==bd]

    def get_reward(self, p):
        if self.current_round==5:
            det = self.get_score(p,'det')
            if det==0: return -3.0
            return 5.0 if p in self.get_final_winner() else 0.5
        return 1.0 if self.get_round_winner()==p else 0.0

    # ── Feature ───────────────────────────────────────────────────
    def _feat(self, p):
        f = np.zeros(FEAT_DIM, np.float32)
        f[0:9]   = self.matrices[p].flatten().astype(np.float32)
        f[9:12]  = self.x_vector.flatten().astype(np.float32)
        f[12]    = float(self.current_round)
        sc = np.array([self.get_score(i) for i in range(6)], np.float32)
        f[13:19] = sc / (np.max(np.abs(sc)) + 1e-8)
        dt = np.array([float(round(np.linalg.det(self.matrices[i].astype(float))))
                       for i in range(6)], np.float32)
        f[19:25] = dt / (np.max(np.abs(dt)) + 1e-8)
        f[25]    = float(sum(1 for s in sc if s > sc[p])) / 5.0
        f[26]    = 1.0
        return f

    # ── Forward / sample ──────────────────────────────────────────
    def _fwd(self, feat, w1, b1, wp, bp, wv, bv):
        h = np.maximum(0, feat @ w1 + b1)
        return h @ wp + bp, h, float((h @ wv + bv)[0])

    def _sample(self, logits, mask, temp=1.0):
        lm = logits + (1 - mask) * (-1e9)
        lm = lm - lm.max()
        pr = np.exp(lm); pr /= pr.sum()
        if temp > 1e-6:
            t  = np.exp(lm / temp); t /= t.sum()
            a  = int(np.random.choice(len(t), p=t))
        else:
            a  = int(np.argmax(lm))
        return a, float(np.log(pr[a] + 1e-8)), pr

    # ── Learner agent ─────────────────────────────────────────────
    def agent_learner(self, p, normal=True, temp=1.0):
        if normal:
            w1,b1,wp,bp,wv,bv = self.w1,self.b1,self.w_pi,self.b_pi,self.w_v,self.b_v
            ad = ACT_DIM
        else:
            w1,b1,wp,bp,wv,bv = self.pw1,self.pb1,self.pw_pi,self.pb_pi,self.pw_v,self.pb_v
            ad = PRIV_DIM
        feat = self._feat(p)
        logits, _, value = self._fwd(feat, w1,b1,wp,bp,wv,bv)
        mask = np.ones(ad, np.float32)
        if normal:
            for r,c in self.used_cells[p]: mask[(r*3+c)*2]=mask[(r*3+c)*2+1]=0.0
        action, lp, _ = self._sample(logits, mask, temp)
        if normal:
            ri,ci = (action//2)//3, (action//2)%3
            self.matrices[p][ri,ci] += 1 if action%2==0 else -1
            self.used_cells[p].add((ri,ci))
        else:
            self.apply_privilege(p, action//2, action%2)
        return feat, action, lp, value, mask

    def agent_snapshot(self, p, snap):
        feat = self._feat(p)
        logits,_,_ = self._fwd(feat,snap['w1'],snap['b1'],snap['w_pi'],snap['b_pi'],snap['w_v'],snap['b_v'])
        mask = np.ones(ACT_DIM, np.float32)
        for r,c in self.used_cells[p]: mask[(r*3+c)*2]=mask[(r*3+c)*2+1]=0.0
        a,_,_ = self._sample(logits, mask, 0.0)
        ri,ci = (a//2)//3, (a//2)%3
        self.matrices[p][ri,ci] += 1 if a%2==0 else -1
        self.used_cells[p].add((ri,ci))

    def agent_random(self, p):
        avail = [(r,c) for r in range(3) for c in range(3) if (r,c) not in self.used_cells[p]]
        if not avail: avail = [(r,c) for r in range(3) for c in range(3)]
        r,c = random.choice(avail)
        self.matrices[p][r,c] += random.choice([-1,1])
        self.used_cells[p].add((r,c))

    def agent_gen3(self, p):
        a = gen3_act(self.matrices, p, self.x_vector, self.current_round, False, self.used_cells[p])
        if a is None: self.agent_random(p); return
        ri,ci = (a//2)//3, (a//2)%3
        self.matrices[p][ri,ci] += 1 if a%2==0 else -1
        self.used_cells[p].add((ri,ci))

    # ── Privilege ─────────────────────────────────────────────────
    def apply_privilege(self, winner, priv_type, scope):
        if scope==0:
            self._do_priv(priv_type, winner, self._best_priv(priv_type, winner))
        else:
            eff = self._best_priv_global(priv_type, winner)
            for t in range(6):
                if t!=winner: self._do_priv(priv_type, t, eff)

    def agent_random_privilege(self, p):
        self.apply_privilege(p, random.randint(0,3), random.randint(0,1))

    def agent_gen3_priv(self, p):
        a = gen3_act(self.matrices, p, self.x_vector, self.current_round, True, set())
        if a is None or a<18: self.agent_random_privilege(p); return
        pi=a-18; scope='self' if pi%2==0 else 'others'; pi//=2
        pairs3=[(0,1),(0,2),(1,2)]
        if   pi<3:  cmd,args='swap_col',pairs3[pi]
        elif pi<6:  cmd,args='swap_row',pairs3[pi-3]
        elif pi<15: cmd,args='zero',((pi-6)//3,(pi-6)%3)
        else:
            ap=[(i,j) for i in range(9) for j in range(i+1,9)]
            cmd,args='swap_elem',ap[pi-15]
        targets=[p] if scope=='self' else [t for t in range(6) if t!=p]
        for t in targets:
            m=self.matrices[t]
            if cmd=='swap_col': c1,c2=args; tmp=m[:,c1].copy(); m[:,c1]=m[:,c2]; m[:,c2]=tmp
            elif cmd=='swap_row': r1,r2=args; tmp=m[r1].copy(); m[r1]=m[r2]; m[r2]=tmp
            elif cmd=='zero': m[args[0],args[1]]=0
            elif cmd=='swap_elem':
                i1,i2=args; r1,c1=i1//3,i1%3; r2,c2=i2//3,i2%3
                m[r1,c1],m[r2,c2]=m[r2,c2],m[r1,c1]

    def _eval(self, m):
        mf=m.astype(float)
        return float(np.sum(mf@self.x_vector.astype(float))) if self.current_round<=2 \
               else float(round(np.linalg.det(mf)))

    def _best_priv(self, pt, ti):
        m=self.matrices[ti]; best,bs=None,-float('inf')
        if pt in [0,1]:
            for i,j in [(0,1),(0,2),(1,2)]:
                n=m.copy()
                if pt==0: tmp=n[i].copy(); n[i]=n[j]; n[j]=tmp
                else:     tmp=n[:,i].copy(); n[:,i]=n[:,j]; n[:,j]=tmp
                s=self._eval(n)
                if s>bs: bs=s; best=(i,j)
        elif pt==2:
            for r in range(3):
                for c in range(3):
                    if m[r,c]==0: continue
                    n=m.copy(); n[r,c]=0; s=self._eval(n)
                    if s>bs: bs=s; best=(r,c)
        elif pt==3:
            pos=[(r,c) for r in range(3) for c in range(3)]
            for k in range(len(pos)):
                for l in range(k+1,len(pos)):
                    r1,c1=pos[k]; r2,c2=pos[l]
                    if m[r1,c1]==m[r2,c2]: continue
                    n=m.copy(); tmp=int(n[r1,c1]); n[r1,c1]=n[r2,c2]; n[r2,c2]=tmp
                    s=self._eval(n)
                    if s>bs: bs=s; best=(r1,c1,r2,c2)
        return best

    def _best_priv_global(self, pt, wi):
        others=[t for t in range(6) if t!=wi]; best,bt=None,float('inf')
        def try_eff(eff):
            nonlocal best,bt
            tot=sum(self._eval(self._priv_copy(pt,t,eff)) for t in others)
            if tot<bt: bt=tot; best=eff
        if pt in [0,1]:
            for pair in [(0,1),(0,2),(1,2)]: try_eff(pair)
        elif pt==2:
            for r in range(3):
                for c in range(3): try_eff((r,c))
        elif pt==3:
            pos=[(r,c) for r in range(3) for c in range(3)]
            for k in range(len(pos)):
                for l in range(k+1,len(pos)): try_eff((pos[k][0],pos[k][1],pos[l][0],pos[l][1]))
        return best

    def _priv_copy(self, pt, ti, eff):
        n=self.matrices[ti].copy(); self._do_priv_on(pt,n,eff); return n

    def _do_priv(self, pt, ti, eff):
        if eff is None: return
        self._do_priv_on(pt, self.matrices[ti], eff)

    def _do_priv_on(self, pt, m, eff):
        if eff is None: return
        if pt==0: i,j=eff; tmp=m[i].copy(); m[i]=m[j]; m[j]=tmp
        elif pt==1: i,j=eff; tmp=m[:,i].copy(); m[:,i]=m[:,j]; m[:,j]=tmp
        elif pt==2: r,c=eff; m[r,c]=0
        elif pt==3: r1,c1,r2,c2=eff; tmp=int(m[r1,c1]); m[r1,c1]=m[r2,c2]; m[r2,c2]=tmp

    # ── PPO update ────────────────────────────────────────────────
    def _ppo_update(self, buf, adv, returns, is_normal):
        n       = len(buf)
        feats   = np.array(buf.feats,     np.float32)
        actions = np.array(buf.actions,   np.int32)
        old_lp  = np.array(buf.log_probs, np.float32)
        masks   = np.array(buf.masks,     np.float32)
        adv_n   = (adv - adv.mean()) / (adv.std() + 1e-8)

        p_slice = slice(0,6) if is_normal else slice(6,12)

        for _ in range(PPO_EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, MINI_BATCH):
                b = idx[start:start+MINI_BATCH]
                if len(b) < 4: continue
                B  = len(b)
                f  = feats[b];   a  = actions[b]
                op = old_lp[b];  av = adv_n[b]
                rt = returns[b]; mk = masks[b]

                if is_normal:
                    w1,b1,wp,bp,wv,bv = self.w1,self.b1,self.w_pi,self.b_pi,self.w_v,self.b_v
                else:
                    w1,b1,wp,bp,wv,bv = self.pw1,self.pb1,self.pw_pi,self.pb_pi,self.pw_v,self.pb_v

                # Forward (vectorized)
                h   = np.maximum(0, f @ w1 + b1)
                lg  = h @ wp + bp
                vs  = (h @ wv + bv).squeeze(-1)

                # Masked softmax
                lm  = lg + (1-mk)*(-1e9)
                lm -= lm.max(1, keepdims=True)
                pr  = np.exp(lm); pr /= pr.sum(1, keepdims=True)
                pr  = np.clip(pr, 1e-8, 1.0)

                # PPO ratio
                nlp = np.log(pr[np.arange(B), a])
                rat = np.exp(np.clip(nlp - op, -10, 10))

                # Clip mask (gradient = 0 where clipped)
                nc = ~((av>0)&(rat>1+CLIP_EPS)) & ~((av<0)&(rat<1-CLIP_EPS))

                # Policy gradient [B, act_dim]
                sc  = (av * rat * nc.astype(np.float32))[:, None]
                ea  = np.zeros_like(pr); ea[np.arange(B), a] = 1.0
                gpi = (ea - pr) * sc

                # Entropy gradient
                gent = ENTROPY_COEF * (-(np.log(pr) + 1.0))

                glogits = gpi + gent
                gval    = ((rt - vs) * VALUE_COEF)[:, None]

                # Backprop through shared layer
                dh = (glogits @ wp.T + gval @ wv.T) * (h > 0)

                grads = [f.T@dh/B, dh.mean(0),
                         h.T@glogits/B, glogits.mean(0),
                         h.T@gval/B,    gval.mean(0)]

                # Grad norm clip
                norm = np.sqrt(sum(np.sum(g**2) for g in grads))
                if norm > MAX_GRAD_NORM:
                    grads = [g*(MAX_GRAD_NORM/(norm+1e-6)) for g in grads]

                # Adam
                self.adam_t += 1
                t   = self.adam_t
                bc1 = 1 - 0.9**t; bc2 = 1 - 0.999**t
                for i, g in enumerate(grads):
                    pi = p_slice.start + i
                    self.m[pi] = 0.9*self.m[pi]   + 0.1*g
                    self.v[pi] = 0.999*self.v[pi]  + 0.001*g*g
                    d = LR*(self.m[pi]/bc1)/(np.sqrt(self.v[pi]/bc2)+1e-8)
                    if is_normal:
                        [self.w1,self.b1,self.w_pi,self.b_pi,self.w_v,self.b_v][i] += d
                    else:
                        [self.pw1,self.pb1,self.pw_pi,self.pb_pi,self.pw_v,self.pb_v][i] += d

    # ── Snapshot ──────────────────────────────────────────────────
    def _snapshot(self):
        return {k: getattr(self,k).copy()
                for k in ['w1','b1','w_pi','b_pi','w_v','b_v']}

    # ── Training loop ─────────────────────────────────────────────
    def run_simulation(self, num_episodes=1_200_000):
        has_g3 = load_gen3_weights()
        print(f"\n학습 시작: {num_episodes:,}회  Gen3: {'ON' if has_g3 else 'OFF'}")
        print("구성: Random 20% / 리그 25% / OldSelf 45% / Gen3 10%\n")

        buf_n = RolloutBuffer(ROLLOUT_NORMAL)
        buf_p = RolloutBuffer(ROLLOUT_PRIV)
        self.learner_history.append(self._snapshot())
        win_cnt = surv_cnt = 0
        best_wr = 0.0
        TEMP_START, TEMP_END = 1.0, 0.05

        try:
            for ep in tqdm(range(num_episodes)):
                if ep % 2000 == 0:
                    self.learner_history.append(self._snapshot())

                temp = TEMP_START - (TEMP_START-TEMP_END)*(ep/num_episodes)

                opponents = []
                for _ in range(5):
                    rv = random.random()
                    if   rv < 0.20: opponents.append("Random")
                    elif rv < 0.45 and len(self.learner_history)>1: opponents.append("League")
                    elif rv < 0.90: opponents.append("OldSelf")
                    else:           opponents.append("Gen3")

                self.reset_game()
                round_rewards = [0.0]*5
                gbuf_n = []   # (feat,act,lp,val,mask,r_num,is_last)
                gbuf_p = []   # (feat,act,lp,val,mask,r_num)

                for r in range(1,6):
                    self.current_round = r
                    for s in self.used_cells: s.clear()

                    for p in range(6):
                        for step in range(r):
                            if p == 0:
                                feat,act,lp,val,mask = self.agent_learner(0, True, temp)
                                gbuf_n.append((feat,act,lp,val,mask,r,step==r-1))
                            else:
                                opp = opponents[p-1]
                                if   opp=="Random":             self.agent_random(p)
                                elif opp=="Gen3" and has_g3:    self.agent_gen3(p)
                                elif opp=="League":             self.agent_snapshot(p, random.choice(self.learner_history))
                                else:                           self.agent_snapshot(p, self.learner_history[-1])

                    self.calculate_x()
                    rw = self.get_round_winner()
                    round_rewards[r-1] = 1.0 if rw==0 else 0.0

                    if rw==0:
                        feat,act,lp,val,mask = self.agent_learner(0, False, temp)
                        gbuf_p.append((feat,act,lp,val,mask,r))
                    elif rw is not None:
                        if has_g3 and opponents[rw-1]=="Gen3": self.agent_gen3_priv(rw)
                        else:                                   self.agent_random_privilege(rw)

                final_r = self.get_reward(0)
                round_rewards[4] += final_r

                # Add to buffer (reward assigned to last action of each round)
                for feat,act,lp,val,mask,r_num,is_last in gbuf_n:
                    reward = round_rewards[r_num-1] if is_last else 0.0
                    buf_n.add(feat,act,lp,val,reward, is_last and r_num==5, mask)

                for feat,act,lp,val,mask,r_num in gbuf_p:
                    buf_p.add(feat,act,lp,val,round_rewards[r_num-1], r_num==5, mask)

                # PPO update when buffers are full
                if buf_n.ready():
                    adv,ret = buf_n.compute_gae()
                    self._ppo_update(buf_n, adv, ret, True)
                    buf_n.reset()
                if buf_p.ready():
                    adv,ret = buf_p.compute_gae()
                    self._ppo_update(buf_p, adv, ret, False)
                    buf_p.reset()

                if sum(round_rewards) > 0: surv_cnt += 1
                if final_r >= 5.0:         win_cnt  += 1

                if ep > 0 and ep % 10000 == 0:
                    cur_wr = win_cnt/ep
                    tqdm.write(f"  ep={ep:,}: 승률={cur_wr*100:.1f}%  생존율={surv_cnt/ep*100:.1f}%")
                    if cur_wr > best_wr:
                        best_wr = cur_wr
                        self._save("strategy_weights_best.json")
                        tqdm.write(f"  → best 저장 ({best_wr*100:.1f}%)")

        except KeyboardInterrupt:
            print("\n[중단] 저장 중...")

        ep_done = max(ep, 1)
        print(f"\n최종: 승률={win_cnt/ep_done*100:.2f}%  생존율={surv_cnt/ep_done*100:.2f}%")
        self._save()

    # ── Save / Load ───────────────────────────────────────────────
    def _save(self, path="strategy_weights.json"):
        data = {}
        for k in ['w1','b1','w_pi','b_pi','w_v','b_v',
                  'pw1','pb1','pw_pi','pb_pi','pw_v','pb_v']:
            data[k] = getattr(self,k).tolist()
        data['w2']=data['w_pi']; data['b2']=data['b_pi']
        data['pw2']=data['pw_pi']; data['pb2']=data['pb_pi']
        data['feat_dim']=FEAT_DIM; data['hidden_dim']=HIDDEN_DIM
        with open(path,'w',encoding='utf-8') as f: json.dump(data,f)
        print(f"저장 완료: {path}")

    def load_weights(self, path="strategy_weights.json"):
        if not os.path.exists(path): return False
        try:
            with open(path,encoding='utf-8') as f: data=json.load(f)
            km={'w1':'w1','b1':'b1','w_pi':['w_pi','w2'],'b_pi':['b_pi','b2'],
                'w_v':'w_v','b_v':'b_v','pw1':'pw1','pb1':'pb1',
                'pw_pi':['pw_pi','pw2'],'pb_pi':['pb_pi','pb2'],'pw_v':'pw_v','pb_v':'pb_v'}
            for attr,keys in km.items():
                keys=[keys] if isinstance(keys,str) else keys
                for k in keys:
                    if k in data: setattr(self,attr,np.array(data[k],np.float32)); break
            all_p=[self.w1,self.b1,self.w_pi,self.b_pi,self.w_v,self.b_v,
                   self.pw1,self.pb1,self.pw_pi,self.pb_pi,self.pw_v,self.pb_v]
            self.m=[np.zeros_like(p) for p in all_p]
            self.v=[np.zeros_like(p) for p in all_p]
            self.adam_t=0
            print("가중치 로드 완료"); return True
        except Exception as e:
            print(f"로드 실패: {e}"); return False

    # ── Interactive ───────────────────────────────────────────────
    def play_interactive(self):
        self.reset_game()
        print("\n"+"="*50+"\n [MATCH] AI (T1) vs YOU (T2) vs Bots\n"+"="*50)
        for r in range(1,6):
            self.current_round=r
            for s in self.used_cells: s.clear()
            print(f"\n{'='*20} ROUND {r} {'='*20}")
            for p in range(6):
                if p==0:
                    for _ in range(r): self.agent_learner(p,True,0.0)
                    print("[AI] 완료.")
                elif p==1:
                    print(f"\n[YOUR TURN]\n{self.matrices[p]}")
                    for a in range(r):
                        while True:
                            try:
                                mv=input(f"행동 {a+1}/{r} (행 열 값): ").split()
                                row,col,val=int(mv[0]),int(mv[1]),int(mv[2])
                                if val not in[-1,1] or not(0<=row<=2 and 0<=col<=2): raise ValueError
                                if (row,col) in self.used_cells[p]: print("이미 사용한 칸"); continue
                                self.matrices[p][row,col]+=val; self.used_cells[p].add((row,col)); break
                            except: print("형식: 행(0-2) 열(0-2) 값(-1/1)")
                else:
                    for _ in range(r): self.agent_random(p)
            self.calculate_x()
            scores=[self.get_score(i) for i in range(6)]
            print(f"X={self.x_vector.flatten()}  점수={[round(s,1) for s in scores]}")
            w=self.get_round_winner()
            if w is not None:
                print(f"라운드 승자: T{w+1}")
                if w==0:
                    _,act,_,_,_=self.agent_learner(0,False,0.0)
                    print(f"[AI 특권] {PRIV_NAMES[act//2]} / {'내팀' if act%2==0 else '전타팀'}")
                elif w!=1: self.agent_random_privilege(w)
        print("\n[최종]")
        for i in range(6):
            d=self.get_score(i,'det'); ax=self.get_score(i,'ax')
            lbl="AI" if i==0 else "YOU" if i==1 else f"Bot{i+1}"
            st="WINNER" if i in self.get_final_winner() else "SURVIVED" if d!=0 else "ELIM"
            print(f" {lbl}: Ax={ax:.1f} det={d} [{st}]")


if __name__ == "__main__":
    game = MatrixGame()
    print("=== Evo-Matrix AI Engine v6.0 ===")
    print("PPO | GAE(λ=0.95) | 벡터화 미니배치 | best 자동 저장")
    print("1. 대전 테스트")
    print("2. 신규 학습 (1,200,000 에피소드)")
    print("3. 이어서 학습")
    choice = input("\n선택 (1/2/3): ").strip()
    if   choice=="1":
        if game.load_weights(): game.play_interactive()
        else: print("[오류] 학습된 가중치 없음.")
    elif choice=="2": game.run_simulation(1_200_000)
    elif choice=="3": game.load_weights(); game.run_simulation(1_200_000)
    else: print("잘못된 선택")
