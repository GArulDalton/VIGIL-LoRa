import numpy as np, pickle, json, time, sys
import evaluate as ev
from nn import masked_softmax
def pol_dqn(path='models/dqn.pkl'):
    q = pickle.load(open(path, 'rb'))['q']
    def f(env, st):
        return np.where(env.mask(), q.forward(env.obs()), -1e9).argmax(1)
    return f
METHODS = {
    'ADR': lambda: ev.pol_adr(), 'EH-ADR': lambda: ev.pol_ehadr(), 'AoI-greedy': lambda: ev.pol_aoi(),
    'VoI-threshold': lambda: ev.pol_voi(), 'Random': lambda: ev.pol_random(), 'DQN': lambda: pol_dqn(),
    'VIGIL-LoRa': lambda: ev.pol_net('models/vigil.pkl'),
}
res = {}
SH = ('VIGIL-LoRa', 'DQN')
def save(): json.dump(res, open('results/results.json', 'w'))
part = sys.argv[1]
t0 = time.time()
if part == 'scale':
    for N in [100, 250, 500, 1000, 2000]:
        for m, mk in METHODS.items():
            for s in range(3):
                r = ev.run(mk(), N=N, seed=s, days=2, shield=(m in SH)); r.pop('delays') if N != 500 else None
                res[f'scale|{m}|{N}|{s}'] = r
            print(N, m, round(time.time()-t0), flush=True); save()
    json.dump(res, open('results/res_scale.json', 'w'))
if part == 'gw':
    for G in [1, 3]:
        for m, mk in METHODS.items():
            for s in range(3):
                r = ev.run(mk(), N=1000, G=G, seed=s, days=2, shield=(m in SH)); r.pop('delays')
                res[f'gw|{m}|{G}|{s}'] = r
            print(G, m, round(time.time()-t0), flush=True)
    json.dump(res, open('results/res_gw.json', 'w'))
if part == 'abl':
    for v in ['vigil', 'no_voi', 'no_lyap', 'no_constr', 'no_central', 'ideal_orth', 'no_shield']:
        for s in range(3):
            r = ev.run(ev.pol_net(f'models/{v}.pkl'), N=500, seed=s, days=2, shield=(v != 'no_shield')); r.pop('delays')
            res[f'abl|{v}|{s}'] = r
        print(v, round(time.time()-t0), flush=True)
    json.dump(res, open('results/res_abl.json', 'w'))
if part == 'trace':
    out = {}
    for m in ['ADR', 'VoI-threshold', 'VIGIL-LoRa']:
        tr = []
        env_kw = dict(N=200, seed=7, days=1)
        ev.trace_i = 5
        ev.run(METHODS[m](), trace=tr, shield=(m in SH), **env_kw); out[m] = tr
    json.dump(out, open('results/res_trace.json', 'w'))
