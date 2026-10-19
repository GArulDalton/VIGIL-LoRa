import json, numpy as np, pickle, time
S = json.load(open('results/res_scale.json')); G = json.load(open('results/res_gw.json')); A = json.load(open('results/res_abl.json'))
M = ['ADR', 'EH-ADR', 'AoI-greedy', 'VoI-threshold', 'Random', 'DQN', 'VIGIL-LoRa']
K = ['nmse', 'outage', 'pdr', 'delay_min', 'missed', 'energy_J_node_day', 'jain', 'tx_per_node_day', 'aoi_min']
def ms(vals): vals = np.array(vals, float); return [float(np.nanmean(vals)), float(np.nanstd(vals))]
out = {'main': {}, 'scale': {}, 'gw': {}, 'abl': {}}
for m in M:
    out['main'][m] = {k: ms([S[f'scale|{m}|500|{s}'][k] for s in range(3)]) for k in K}
    out['scale'][m] = {str(n): {k: ms([S[f'scale|{m}|{n}|{s}'][k] for s in range(3)]) for k in K} for n in [100, 250, 500, 1000, 2000]}
    out['gw'][m] = {str(g): {k: ms([G[f'gw|{m}|{g}|{s}'][k] for s in range(3)]) for k in K} for g in [1, 3]}
for v in ['vigil', 'no_voi', 'no_lyap', 'no_constr', 'no_central', 'ideal_orth', 'no_shield']:
    out['abl'][v] = {k: ms([A[f'abl|{v}|{s}'][k] for s in range(3)]) for k in K}
d = pickle.load(open('models/vigil.pkl', 'rb')); a = d['actor']
out['params'] = int(a.n_params())
out['macs'] = int(sum(W.size for W in a.W))
x = np.random.rand(1, 12).astype(np.float32); t = time.perf_counter()
for _ in range(2000): a.forward(x)
out['infer_us_cpu'] = (time.perf_counter()-t)/2000*1e6
out['final_lambda'] = d['lam']; out['hist_last'] = d['hist'][-10:]
json.dump(out, open('results/summary.json', 'w'), indent=1)
for m in M: print(m, {k: round(v[0], 3) for k, v in out['main'][m].items()})
print('ABL'); [print(v, {k: round(x[0], 3) for k, x in r.items()}) for v, r in out['abl'].items()]
print('GW'); [print(m, out['gw'][m]['1']['nmse'][0], out['gw'][m]['3']['nmse'][0], out['gw'][m]['1']['pdr'][0], out['gw'][m]['3']['pdr'][0]) for m in M]
print('N2000', {m: round(out['scale'][m]['2000']['nmse'][0], 3) for m in M})
print(out['params'], out['macs'], out['infer_us_cpu'])
