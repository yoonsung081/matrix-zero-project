"""
Matrix Game Battle Simulator
battle.html 게임 로직을 Python으로 포팅 — Gen3 vs My AI 대결 시뮬레이션 후 JSON 출력
"""
import json
import math
import random
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
TEAM_NAMES = ['T1 Gen3', 'T2 My AI', 'T3 Bot', 'T4 Bot', 'T5 Bot', 'T6 Bot']
TEAM_TYPES = ['gen3', 'myai', 'bot', 'bot', 'bot', 'bot']


# ─── 가중치 로드 ─────────────────────────────────────────────────
def load_weights():
    gen3w = None
    try:
        with open(BASE_DIR / 'gen3_weights.json', encoding='utf-8') as f:
            gen3w = json.load(f)
        print("[OK] gen3_weights.json loaded")
    except Exception as e:
        print(f"[WARN] gen3_weights.json load failed ({e}) - Gen3 uses dummy output")
    return gen3w


# ─── 수학 헬퍼 ───────────────────────────────────────────────────
def det(m):
    return (m[0]*(m[4]*m[8]-m[5]*m[7])
           -m[1]*(m[3]*m[8]-m[5]*m[6])
           +m[2]*(m[3]*m[7]-m[4]*m[6]))

def calc_x(mats):
    return [math.floor(sum(m[i*4] for m in mats) / 6) for i in range(3)]

def ax_sum(m, x):
    return sum(m[r*3+c] * x[c] for r in range(3) for c in range(3))

def score(m, x, rnd):
    return ax_sum(m, x) if rnd <= 2 else det(m)


# ─── Gen3 추론 ───────────────────────────────────────────────────
def gen3_obs(t_idx, matrices, x_vec, rnd, is_priv):
    obs = list(matrices[t_idx])
    for p in range(6):
        if p == t_idx:
            continue
        m = list(matrices[p])
        m[0] = m[4] = m[8] = 0  # 대각선 마스킹
        obs.extend(m)
    obs.extend(x_vec)
    obs.append(rnd)
    obs.append(1 if is_priv else 0)
    return obs  # len=59

def gen3_fwd(obs, gen3w):
    if gen3w is None:
        return [0.0] * 120
    w1, b1 = gen3w['pn_w1'], gen3w['pn_b1']  # w1: [64][59]
    w2, b2 = gen3w['pn_w2'], gen3w['pn_b2']  # w2: [64][64]
    wa, ba = gen3w['act_w'], gen3w['act_b']   # wa: [120][64]
    h1 = [math.tanh(b1[j] + sum(obs[i]*w1[j][i] for i in range(59))) for j in range(len(b1))]
    h2 = [math.tanh(b2[j] + sum(h1[i]*w2[j][i] for i in range(len(h1)))) for j in range(len(b2))]
    return [ba[j] + sum(h2[i]*wa[j][i] for i in range(len(h2))) for j in range(len(ba))]

def gen3_mask(used_idx, is_priv):
    m = [False] * 120
    if is_priv:
        for i in range(18, 120):
            m[i] = True
    else:
        for i in range(18):
            if i // 2 not in used_idx:
                m[i] = True
    return m

def gen3_pick(obs, mask, gen3w):
    logits = gen3_fwd(obs, gen3w)
    masked = [l if mask[i] else -math.inf for i, l in enumerate(logits)]
    return masked.index(max(masked))

def gen3_decode_priv(act_idx):
    p = act_idx - 18
    scope = 'self' if p % 2 == 0 else 'others'
    p //= 2
    pairs3 = [[0,1],[0,2],[1,2]]
    if p < 3:  return {'cmd': 'swap_col',  'scope': scope, 'args': pairs3[p]}
    p -= 3
    if p < 3:  return {'cmd': 'swap_row',  'scope': scope, 'args': pairs3[p]}
    p -= 3
    if p < 9:  return {'cmd': 'zero',      'scope': scope, 'args': [p//3, p%3]}
    p -= 9
    pairs = [[i,j] for i in range(9) for j in range(i+1,9)]
    return {'cmd': 'swap_elem', 'scope': scope, 'args': pairs[p]}

def apply_gen3_priv(winner_idx, priv, matrices):
    targets = [winner_idx] if priv['scope'] == 'self' else [i for i in range(6) if i != winner_idx]
    for t in targets:
        m = list(matrices[t])
        cmd, args = priv['cmd'], priv['args']
        if cmd == 'swap_col':
            c1, c2 = args
            for r in range(3): m[r*3+c1], m[r*3+c2] = m[r*3+c2], m[r*3+c1]
        elif cmd == 'swap_row':
            r1, r2 = args
            for c in range(3): m[r1*3+c], m[r2*3+c] = m[r2*3+c], m[r1*3+c]
        elif cmd == 'zero':
            m[args[0]*3+args[1]] = 0
        else:
            m[args[0]], m[args[1]] = m[args[1]], m[args[0]]
        matrices[t] = m


# ─── My AI 스마트 휴리스틱 ───────────────────────────────────────
def ax_score(m, x_vec):
    return sum(m[i] * x_vec[i % 3] for i in range(9))

def my_pick(t_idx, matrices, x_vec, rnd, used_idx):
    my_m = list(matrices[t_idx])
    used_set = set(used_idx)
    best = {'cell_idx': 0, 'delta': 1, 'score': -math.inf}
    for i in range(9):
        if i in used_set:
            continue
        for delta in [1, -1]:
            n = list(my_m); n[i] += delta
            d  = det(n)
            ax = ax_score(n, x_vec)
            if rnd <= 2:
                s = ax * 1000 + (5 if i in (0,4,8) else 0)
            elif rnd <= 4:
                s = -5000 if d == 0 else d * 100 + ax * 2
            else:
                s = -100000 if d == 0 else ax * 1000 + abs(d) * 5
            if s > best['score']:
                best = {'cell_idx': i, 'delta': delta, 'score': s}
    return best


# ─── 특권 적용 ───────────────────────────────────────────────────
def apply_best_priv(t_idx, matrices, x_vec, rnd, priv_type, scope):
    minimize = (scope == 'others')
    sf = lambda m: score(m, x_vec, rnd)
    m = list(matrices[t_idx])

    def pick_best_pair(candidates_fn):
        best_s = math.inf if minimize else -math.inf
        best_args = None
        for args, n in candidates_fn(m):
            s = sf(n)
            if (minimize and s < best_s) or (not minimize and s > best_s):
                best_s, best_args = s, args
        return best_args

    if priv_type == 'swap_row':
        def rows(m):
            for a in range(3):
                for b in range(a+1,3):
                    n=list(m)
                    for c in range(3): n[a*3+c],n[b*3+c]=n[b*3+c],n[a*3+c]
                    yield (a,b),n
        a,b = pick_best_pair(rows)
        for c in range(3): m[a*3+c],m[b*3+c]=m[b*3+c],m[a*3+c]

    elif priv_type == 'swap_col':
        def cols(m):
            for a in range(3):
                for b in range(a+1,3):
                    n=list(m)
                    for r in range(3): n[r*3+a],n[r*3+b]=n[r*3+b],n[r*3+a]
                    yield (a,b),n
        a,b = pick_best_pair(cols)
        for r in range(3): m[r*3+a],m[r*3+b]=m[r*3+b],m[r*3+a]

    elif priv_type == 'zero':
        def zeros(m):
            for i in range(9):
                if m[i] != 0:
                    n=list(m); n[i]=0
                    yield (i,),n
        (idx,) = pick_best_pair(zeros)
        m[idx] = 0

    else:  # swap_elements
        def swaps(m):
            for a in range(9):
                for b in range(a+1,9):
                    n=list(m); n[a],n[b]=n[b],n[a]
                    yield (a,b),n
        a,b = pick_best_pair(swaps)
        m[a],m[b] = m[b],m[a]

    matrices[t_idx] = m

def my_priv(t_idx, matrices, x_vec, rnd):
    types = ['swap_row', 'swap_col', 'zero', 'swap_elements']
    best = {'type': 'swap_row', 'scope': 'self', 'gain': -math.inf}
    my_before = score(matrices[t_idx], x_vec, rnd)

    for priv_type in types:
        for scope in ['self', 'others']:
            mc = [list(m) for m in matrices]
            targets = [t_idx] if scope == 'self' else [i for i in range(6) if i != t_idx]
            for t in targets:
                apply_best_priv(t, mc, x_vec, rnd, priv_type, scope)
            if scope == 'self':
                gain = score(mc[t_idx], x_vec, rnd) - my_before
            else:
                before = sum(score(matrices[i], x_vec, rnd) for i in range(6) if i != t_idx)
                after  = sum(score(mc[i],       x_vec, rnd) for i in range(6) if i != t_idx)
                gain = before - after
            if gain > best['gain']:
                best = {'type': priv_type, 'scope': scope, 'gain': gain}

    targets = [t_idx] if best['scope'] == 'self' else [i for i in range(6) if i != t_idx]
    for t in targets:
        apply_best_priv(t, matrices, x_vec, rnd, best['type'], best['scope'])
    return {'priv_type': best['type'], 'scope': best['scope']}


# ─── Bot 휴리스틱 ────────────────────────────────────────────────
def heuristic_logits(my_m, x_vec, rnd):
    logits = []
    for i in range(9):
        for delta in [1, -1]:
            n = list(my_m); n[i] += delta
            logits.append(score(n, x_vec, rnd))
    return logits

def bot_pick(t_idx, matrices, x_vec, rnd, used_idx, temp=0.3):
    logits = heuristic_logits(matrices[t_idx], x_vec, rnd)
    for idx in used_idx:
        logits[idx*2] = -1e9; logits[idx*2+1] = -1e9
    valid_max = max((l for l in logits if l > -1e8), default=0)
    exps = [math.exp((l - valid_max) / temp) if l > -1e8 else 0.0 for l in logits]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    r = random.random(); cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            return {'cell_idx': i // 2, 'delta': 1 if i % 2 == 0 else -1}
    return {'cell_idx': 0, 'delta': 1}

def bot_priv(winner_idx, matrices, x_vec, rnd):
    types = ['swap_row', 'swap_col', 'zero', 'swap_elements']
    priv_type = random.choice(types)
    scope = random.choice(['self', 'others'])
    targets = [winner_idx] if scope == 'self' else [i for i in range(6) if i != winner_idx]
    for t in targets:
        apply_best_priv(t, matrices, x_vec, rnd, priv_type, scope)
    return {'priv_type': priv_type, 'scope': scope}


# ─── 승자 판정 ───────────────────────────────────────────────────
def find_winner(matrices, x_vec, rnd):
    scores = sorted([(score(m, x_vec, rnd), i) for i, m in enumerate(matrices)], reverse=True)
    for k in range(min(3, 6)):
        s = scores[k][0]
        if sum(1 for sc, _ in scores if sc == s) == 1:
            return next(i for sc, i in scores if sc == s)
    return None

def final_winner(matrices, x_vec):
    survivors = [(i, det(m), ax_sum(m, x_vec)) for i, m in enumerate(matrices) if det(m) != 0]
    if not survivors:
        return -1
    survivors.sort(key=lambda t: (-t[2], -abs(t[1])))
    return survivors[0][0]


# ─── 게임 1판 시뮬레이션 ─────────────────────────────────────────
def run_game(gen3w, game_num=1, record_detail=True):
    matrices = [[0]*9 for _ in range(6)]
    round_wins = [0] * 6
    rounds_log = []

    for rnd in range(1, 6):
        used_idx = [set() for _ in range(6)]
        actions_log = []

        for act in range(rnd):
            moves = []
            for t in range(6):
                x_vec = calc_x(matrices)
                if t == 0:
                    obs  = gen3_obs(t, matrices, x_vec, rnd, False)
                    mask = gen3_mask(used_idx[t], False)
                    a    = gen3_pick(obs, mask, gen3w)
                    cell_idx = a // 2; delta = 1 if a % 2 == 0 else -1
                elif t == 1:
                    res = my_pick(t, matrices, x_vec, rnd, used_idx[t])
                    cell_idx = res['cell_idx']; delta = res['delta']
                else:
                    res = bot_pick(t, matrices, x_vec, rnd, used_idx[t])
                    cell_idx = res['cell_idx']; delta = res['delta']

                matrices[t][cell_idx] += delta
                used_idx[t].add(cell_idx)
                if record_detail:
                    moves.append({
                        'team': TEAM_NAMES[t],
                        'row': cell_idx // 3 + 1,
                        'col': cell_idx % 3 + 1,
                        'delta': delta
                    })
            if record_detail:
                actions_log.append({'action': act + 1, 'moves': moves})

        x_vec = calc_x(matrices)
        winner_idx = find_winner(matrices, x_vec, rnd)
        priv_log = None

        if winner_idx is not None:
            round_wins[winner_idx] += 1
            if winner_idx == 0:
                obs  = gen3_obs(winner_idx, matrices, x_vec, rnd, True)
                mask = gen3_mask(set(), True)
                a    = gen3_pick(obs, mask, gen3w)
                priv = gen3_decode_priv(a)
                apply_gen3_priv(winner_idx, priv, matrices)
                priv_log = {'cmd': priv['cmd'], 'scope': priv['scope']}
            elif winner_idx == 1:
                res = my_priv(winner_idx, matrices, x_vec, rnd)
                priv_log = {'cmd': res['priv_type'], 'scope': res['scope']}
            else:
                res = bot_priv(winner_idx, matrices, x_vec, rnd)
                priv_log = {'cmd': res['priv_type'], 'scope': res['scope']}

        x_after = calc_x(matrices)
        round_entry = {
            'round': rnd,
            'winner': TEAM_NAMES[winner_idx] if winner_idx is not None else None,
            'privilege': priv_log,
            'x_vec': list(x_after),
        }
        if record_detail:
            round_entry['actions'] = actions_log
            round_entry['state'] = {
                TEAM_NAMES[i]: {
                    'matrix': list(matrices[i]),
                    'det': det(matrices[i]),
                    'ax': round(ax_sum(matrices[i], x_after), 2),
                } for i in range(6)
            }
        rounds_log.append(round_entry)

    x_final = calc_x(matrices)
    fw = final_winner(matrices, x_final)

    return {
        'game': game_num,
        'final_winner': TEAM_NAMES[fw] if fw >= 0 else '전원탈락',
        'round_wins': {TEAM_NAMES[i]: round_wins[i] for i in range(6)},
        'final_state': {
            TEAM_NAMES[i]: {
                'matrix': list(matrices[i]),
                'det': det(matrices[i]),
                'ax': round(ax_sum(matrices[i], x_final), 2),
                'survived': det(matrices[i]) != 0,
            } for i in range(6)
        },
        'x_final': list(x_final),
        'rounds': rounds_log,
    }


# ─── 메인 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Matrix Game Battle Simulator')
    parser.add_argument('-n', '--num_games', type=int, default=1,
                        help='시뮬레이션 게임 수 (기본: 1)')
    parser.add_argument('-o', '--output', type=str, default='battle_log.json',
                        help='출력 JSON 파일 경로 (기본: battle_log.json)')
    parser.add_argument('--seed', type=int, default=None, help='랜덤 시드')
    parser.add_argument('--no-detail', action='store_true',
                        help='액션 상세 로그 생략 (n>100일 때 권장)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    gen3w = load_weights()
    record_detail = not args.no_detail and args.num_games <= 20

    print(f"시뮬레이션 시작: {args.num_games}판")
    games = []
    win_counts = {name: 0 for name in TEAM_NAMES}
    elimination = 0

    for i in range(args.num_games):
        result = run_game(gen3w, game_num=i+1, record_detail=record_detail)
        games.append(result)
        fw = result['final_winner']
        if fw == '전원탈락':
            elimination += 1
        else:
            win_counts[fw] += 1
        if args.num_games >= 10 and (i+1) % max(1, args.num_games//10) == 0:
            pct = (i+1)/args.num_games*100
            print(f"  {i+1:>5}/{args.num_games}  ({pct:.0f}%)")

    n = args.num_games
    win_rates = {k: round(v/n*100, 1) for k, v in win_counts.items()}

    output = {
        'summary': {
            'total_games': n,
            'win_counts': win_counts,
            'win_rates': win_rates,
            'elimination_games': elimination,
        },
        'games': games,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== {n}판 결과 ===")
    for name in TEAM_NAMES:
        rate = win_rates[name]
        bar  = '#' * int(rate / 2)
        print(f"  {name:12s}: {rate:5.1f}%  {bar}")
    if elimination:
        print(f"  전원탈락     : {elimination}판")
    print(f"\n로그 저장 완료 → {args.output}")


if __name__ == '__main__':
    main()
