import numpy as np, pickle
from env import LoRaEHEnv, N_ACT, AIR_TABLE, act_decode
from nn import masked_softmax

def pol_adr(period=8, shield=False):
    def f(env, st):
        if 'ph' not in st: st['ph'] = env.rng.integers(0, period, env.N)
        a = np.where((env.t - st['ph']) % period == 0, env.adr_act, 0)
        return a
    return f
def pol_ehadr(period=8, th=0.4):
    def f(env, st):
        if 'ph' not in st: st['ph'] = env.rng.integers(0, period, env.N)
        due = (env.t - st['ph']) % period == 0
        return np.where(due & (env.E/env.E_max > th), env.adr_act, 0)
    return f
def pol_aoi(th=8, eth=0.3):
    def f(env, st):
        return np.where((env.aoi >= th) & (env.E/env.E_max > eth), env.adr_act, 0)
    return f
def pol_voi(tau=1.0, eth=0.3):
    def f(env, st):
        return np.where((env.err2() > tau) & (env.E/env.E_max > eth), env.adr_act, 0)
    return f
def pol_random(p=1/8):
    def f(env, st):
        m = env.mask(); a = np.zeros(env.N, int)
        go = env.rng.random(env.N) < p
        for i in np.where(go)[0]:
            c = np.where(m[i, 1:])[0]
            if len(c): a[i] = 1 + env.rng.choice(c)
        return a
    return f
def pol_net(path, greedy=False):
    d = pickle.load(open(path, 'rb')); actor = d['actor']; cfg = d['cfg']
    def f(env, st):
        o = env.obs()
        if not cfg['use_voi']: o[:, 0] = 0; o[:, 10] = 0
        if not cfg['use_lyap']: o[:, 7] = 0
        p = masked_softmax(actor.forward(o), env.mask())
        if greedy: return p.argmax(1)
        return (p.cumsum(1) > env.rng.random((env.N, 1))).argmax(1)
    return f

def run(policy, N=500, G=1, seed=0, days=3, shield=True, ideal=False, R=3000, warm=1, trace=None):
    env = LoRaEHEnv(N=N, G=G, R=R, seed=1000+seed, shield=shield, ideal_orth=ideal)
    st = {}
    S = env.slots_day
    for t in range(warm*S):          # warm-up day (not scored)
        env.step(policy(env, st))
    env.events = []
    e2s, outs, tx, succ, aoi, ener = [], [], 0, 0, np.zeros(N), []
    pernode = np.zeros(N); sfc = np.zeros(6)
    for t in range(days*S):
        a = policy(env, st)
        e2, out, info = env.step(a)
        e2s.append(e2.mean()); outs.append(out.mean()); pernode += e2
        tx += info['tx'].sum(); succ += info['succ'].sum(); aoi += env.aoi
        ener.append(info['etx'].sum())
        for s in range(6): sfc[s] += (info['tx'] & (info['sfi'] == s)).sum()
        if trace is not None and t < S:
            trace.append(dict(x=float(env.x_true()[trace_i]), xh=float(env.xhat()[trace_i]),
                              E=float(env.E[trace_i]), tx=bool(info['tx'][trace_i]), ok=bool(info['succ'][trace_i]),
                              h=env.hour()))
    ev = np.array(env.events)
    det = ev[ev >= 0]*env.dt/60.0
    acc = 1/(1+pernode/(days*S))
    jain = acc.sum()**2/(N*(acc**2).sum())
    return dict(nmse=float(np.mean(e2s)), outage=float(np.mean(outs)), pdr=float(succ/max(tx, 1)),
                tx_per_node_day=float(tx/N/days), aoi_min=float(np.median(aoi/(days*S))*env.dt/60),
                delay_min=float(det.mean()) if len(det) else float('nan'),
                missed=float((ev < 0).mean()) if len(ev) else float('nan'),
                energy_J_node_day=float(np.sum(ener)/N/days), jain=float(jain),
                sf_frac=(sfc/max(sfc.sum(), 1)).tolist(), delays=det.tolist())
trace_i = 0
if __name__ == '__main__':
    import time
    for name, p in [('ADR', pol_adr()), ('EH-ADR', pol_ehadr()), ('AoI', pol_aoi()), ('VoI-th', pol_voi()), ('Random', pol_random())]:
        t = time.time(); r = run(p, N=200, days=2)
        print(name, {k: (round(v, 3) if isinstance(v, float) else '') for k, v in r.items() if k not in ('delays', 'sf_frac')}, round(time.time()-t, 1))
    r = run(pol_net('models/vigil.pkl'), N=200, days=2)
    print('VIGIL', {k: (round(v, 3) if isinstance(v, float) else '') for k, v in r.items() if k not in ('delays', 'sf_frac')})
