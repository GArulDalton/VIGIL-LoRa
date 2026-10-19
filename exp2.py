"""New experiments: conformal reserve calibration, paired significance tests, runtime/scalability."""
import numpy as np, json, time, sys, pickle
from scipy.stats import wilcoxon
import evaluate as ev
from env import LoRaEHEnv
from nn import masked_softmax
sys.path.insert(0, '.')
from experiments_lib import pol_dqn

def make_env(N, seed, kmax, knoise, mu):
    env = LoRaEHEnv(N=N, R=3000, seed=1000+seed, shield=True)
    if kmax > 1.0:
        env.kappa = np.random.default_rng(5000+seed).uniform(0.8, kmax, N)
        env.knoise = knoise
    env.sm = mu
    return env

def cal_scores(kmax, knoise, n_net=4, N=50, seed0=700):
    """Calibration: idle overnight drain ratio measured by nodes of separate networks (one night each)."""
    sc = []
    for k in range(n_net):
        rng = np.random.default_rng(5000+seed0+k)
        kap = rng.uniform(0.8, kmax, N) if kmax > 1.0 else np.ones(N)
        noise = np.random.default_rng(9999+seed0+k).uniform(1-knoise, 1+knoise, N) if kmax > 1.0 else np.ones(N)
        sc.append(kap*noise)
    return np.concatenate(sc)

def conformal_mu(scores, eps):
    n = len(scores); k = int(np.ceil((1-eps)*(n+1)))
    return float(np.sort(scores)[min(k, n)-1])

def night_eval(policy, env, days=3, warm=1):
    S = env.slots_day; st = {}
    for t in range(warm*S): env.step(policy(env, st))
    e2s, outs = [], []; nights = []; cur = None
    dusk = int(18*3600/env.dt); dawn = int(6.5*3600/env.dt)
    for t in range(days*S):
        slot = env.t % S
        if slot == dusk:
            cur = dict(ok=env.alive & (env.E >= env.reserve()), died=np.zeros(env.N, bool))
        a = policy(env, st); e2, out, info = env.step(a)
        e2s.append(e2.mean()); outs.append(out.mean())
        if cur is not None:
            cur['died'] |= out
            if env.t % S == dawn:
                nights.append(cur); cur = None
    okd = np.concatenate([n['ok'] for n in nights]); died = np.concatenate([n['died'] for n in nights])
    allnights = died.mean()
    return dict(nmse=float(np.mean(e2s)), outage=float(np.mean(outs)),
                cond_night_outage=float(died[okd].mean()) if okd.any() else float('nan'),
                frac_qualified=float(okd.mean()), night_outage=float(allnights))

part = sys.argv[1]; R = {}
if part == 'conformal':
    eps = 0.05
    for kmax in [1.0, 1.5, 2.0, 2.5]:
        kn = 0.1 if kmax > 1 else 0.0
        mu_hat = conformal_mu(cal_scores(kmax, kn), eps) if kmax > 1 else 1.0
        for name, mu in [('model', 1.2), ('conformal', mu_hat)]:
            rs = [night_eval(ev.pol_net('models/vigil.pkl'), make_env(500, s, kmax, kn, mu)) for s in range(3)]
            R[f'{kmax}|{name}'] = {k: [float(np.nanmean([r[k] for r in rs])), float(np.nanstd([r[k] for r in rs]))] for k in rs[0]}
            R[f'{kmax}|{name}']['mu'] = mu
            print(kmax, name, round(mu, 3), {k: round(v[0], 4) for k, v in R[f'{kmax}|{name}'].items() if k != 'mu'}, flush=True)
        # calibration-set size effect at kmax=2.0
    for ncal in [20, 50, 100, 200, 500]:
        mus = []
        for d in range(200):
            sc = cal_scores(2.0, 0.1, n_net=10, N=50, seed0=10000+d*7)
            rng = np.random.default_rng(d); sc = rng.choice(sc, ncal, replace=False)
            mus.append(conformal_mu(sc, eps))
        test = cal_scores(2.0, 0.1, n_net=40, N=50, seed0=900000)
        viol = [float((test > m).mean()) for m in mus]
        R[f'ncal|{ncal}'] = dict(mean_viol=float(np.mean(viol)), p90_viol=float(np.percentile(viol, 90)), mu_mean=float(np.mean(mus)), mu_std=float(np.std(mus)))
        print('ncal', ncal, R[f'ncal|{ncal}'], flush=True)
    json.dump(R, open('results/res_conformal.json', 'w'), indent=1)

if part == 'wilcoxon':
    M = {'ADR': lambda: ev.pol_adr(), 'EH-ADR': lambda: ev.pol_ehadr(), 'AoI-greedy': lambda: ev.pol_aoi(),
         'VoI-threshold': lambda: ev.pol_voi(), 'Random': lambda: ev.pol_random(), 'DQN': lambda: pol_dqn(),
         'VIGIL-LoRa': lambda: ev.pol_net('models/vigil.pkl')}
    SH = ('VIGIL-LoRa', 'DQN'); seeds = range(10, 20)
    for m, mk in M.items():
        R[m] = [ev.run(mk(), N=500, seed=s, days=2, shield=(m in SH))['nmse'] for s in seeds]
        print(m, np.round(R[m], 3), flush=True)
    W = {}
    v = np.array(R['VIGIL-LoRa'])
    for m in M:
        if m == 'VIGIL-LoRa': continue
        b = np.array(R[m]); d = b - v
        W[m] = dict(mean_red=float(d.mean()), ci=[float(d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d))), float(d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(d)))],
                    wins=float((d > 0).mean()), p=float(wilcoxon(b, v, alternative='greater').pvalue))
        print(m, W[m], flush=True)
    json.dump(dict(raw=R, test=W), open('results/res_wilcoxon.json', 'w'), indent=1)

if part == 'runtime':
    d = pickle.load(open('models/vigil.pkl', 'rb')); actor = d['actor']
    out = {}
    x = np.random.rand(1, 12).astype(np.float32); m = np.ones((1, 19), bool)
    t = time.perf_counter()
    for _ in range(5000): masked_softmax(actor.forward(x), m)
    out['node_decision_us'] = (time.perf_counter()-t)/5000*1e6
    for N in [100, 250, 500, 1000, 2000]:
        env = LoRaEHEnv(N=N, R=3000, seed=1); pol = ev.pol_net('models/vigil.pkl'); st = {}
        tp = ts = 0.0
        for k in range(200):
            a0 = time.perf_counter(); a = pol(env, st); a1 = time.perf_counter(); env.step(a); a2 = time.perf_counter()
            tp += a1-a0; ts += a2-a1
        out[f'N{N}'] = dict(policy_ms_per_slot=tp/200*1e3, per_node_us=tp/200/N*1e6, channel_ms_per_slot=ts/200*1e3)
        print(N, out[f'N{N}'], flush=True)
    out['params'] = int(actor.n_params()); out['macs'] = int(sum(W.size for W in actor.W))
    json.dump(out, open('results/res_runtime.json', 'w'), indent=1)
