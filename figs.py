import json, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial', 'DejaVu Sans'], 'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 200, 'savefig.dpi': 600})
S = json.load(open('results/res_scale.json')); G = json.load(open('results/res_gw.json')); A = json.load(open('results/res_abl.json'))
M = ['ADR', 'EH-ADR', 'AoI-greedy', 'VoI-threshold', 'Random', 'DQN', 'VIGIL-LoRa']
mk = dict(zip(M, ['o', 's', '^', 'v', 'x', 'D', '*'])); col = dict(zip(M, ['C0','C1','C2','C4','C7','C5','C3'])); Ns = [100, 250, 500, 1000, 2000]
def agg(D, pre, m, x, k):
    v = [D[f'{pre}|{m}|{x}|{s}'][k] for s in range(3)]; return np.nanmean(v), np.nanstd(v)
def lineplot(k, yl, fn, logy=False):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in M:
        mu = [agg(S, 'scale', m, n, k) for n in Ns]
        ax.errorbar(Ns, [a for a, b in mu], yerr=[b for a, b in mu], marker=mk[m], color=col[m], ms=5 if m=='VIGIL-LoRa' else 4, lw=1.6 if m == 'VIGIL-LoRa' else 1, capsize=2, label=m)
    ax.set_xscale('log'); ax.set_xticks(Ns); ax.set_xticklabels(Ns); ax.set_xlabel('Number of sensor nodes $N$'); ax.set_ylabel(yl)
    if logy: ax.set_yscale('log')
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(f'fig/{fn}.png'); plt.close(fig)
import os; os.makedirs('fig', exist_ok=True)
lineplot('nmse', 'Normalized estimation MSE', 'fig3_nmse_vs_N')
lineplot('pdr', 'Packet delivery ratio', 'fig4b_pdr_vs_N')
lineplot('outage', 'Energy-outage probability', 'fig4a_outage_vs_N')
lineplot('delay_min', 'Mean anomaly-detection delay (min)', 'fig5b_delay_vs_N')
# convergence
h = pickle.load(open('models/vigil.pkl', 'rb'))['hist']
it = [x['it'] for x in h]; sm = lambda y: np.convolve(y, np.ones(5)/5, mode='valid')
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
for a, k, t in zip(ax, ['nmse', 'outage', 'lam'], ['Training NMSE', 'Outage rate', r'Lagrange multiplier $\lambda$']):
    y = [x[k] for x in h]; a.plot(it, y, alpha=0.3, color='C0'); a.plot(it[4:], sm(y), color='C0'); a.set_title(t, fontsize=9); a.set_xlabel('PPO iteration')
fig.tight_layout(); fig.savefig('fig/fig2_convergence.png'); plt.close(fig)
# delay CDF (N=500)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for m in M:
    d = np.concatenate([S[f'scale|{m}|500|{s}']['delays'] for s in range(3)])
    if len(d): ax.plot(np.sort(d), np.arange(1, len(d)+1)/len(d), label=m, color=col[m], lw=2 if m == 'VIGIL-LoRa' else 1)
ax.set_xlabel('Anomaly-detection delay (min)'); ax.set_ylabel('Empirical CDF'); ax.set_xlim(0, 60); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig('fig/fig5a_delay_cdf.png'); plt.close(fig)
# SF distribution
fig, ax = plt.subplots(figsize=(3.5, 2.4)); w = 0.25
for i, m in enumerate(['ADR', 'VoI-threshold', 'VIGIL-LoRa']):
    f = np.mean([S[f'scale|{m}|500|{s}']['sf_frac'] for s in range(3)], 0)
    ax.bar(np.arange(7, 13)+(i-1)*w, f, w, label=m)
ax.set_xlabel('Spreading factor'); ax.set_ylabel('Fraction of transmissions'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig/fig7_sf_dist.png'); plt.close(fig)
# trade-off energy vs NMSE
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for m in M:
    e = agg(S, 'scale', m, 500, 'energy_J_node_day')[0]; n = agg(S, 'scale', m, 500, 'nmse')[0]
    ax.scatter(e, n, marker=mk[m], color=col[m], s=40 if m == 'VIGIL-LoRa' else 25, label=m)
ax.set_xlabel('Tx energy per node per day (J)'); ax.set_ylabel('NMSE'); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig('fig/fig9_tradeoff.png'); plt.close(fig)
# ablation
V = ['vigil', 'no_voi', 'no_lyap', 'no_constr', 'no_central', 'ideal_orth', 'no_shield']
L = ['Full', 'w/o VoI', 'w/o Lyap.', 'w/o constr.', 'w/o central\ncritic', 'ideal-orth.\ntrained', 'w/o shield']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
for a, k, t in zip(ax, ['nmse', 'outage'], ['NMSE', 'Energy-outage probability']):
    mu = [np.mean([A[f'abl|{v}|{s}'][k] for s in range(3)]) for v in V]; sd = [np.std([A[f'abl|{v}|{s}'][k] for s in range(3)]) for v in V]
    a.bar(range(len(V)), mu, yerr=sd, color=['C3']+['C0']*6, capsize=2); a.set_xticks(range(len(V))); a.set_xticklabels(L, fontsize=6.5); a.set_title(t, fontsize=9)
fig.tight_layout(); fig.savefig('fig/fig8_ablation.png'); plt.close(fig)
# trace
T = json.load(open('results/res_trace.json'))
fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True)
hh = np.arange(len(T['ADR']))*2/60
ax[0].plot(hh, [x['x'] for x in T['VIGIL-LoRa']], 'k', lw=1, label='True process')
for m, c in [('ADR', 'C1'), ('VIGIL-LoRa', 'C3')]:
    ax[0].plot(hh, [x['xh'] for x in T[m]], c, lw=1, label=f'Gateway estimate ({m})')
    ax[1].plot(hh, [x['E'] for x in T[m]], c, lw=1, label=m)
ax[0].set_ylabel('Sensed value (°C)'); ax[0].legend(fontsize=6); ax[1].set_ylabel('Stored energy (J)'); ax[1].set_xlabel('Time of day (h)'); ax[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig('fig/fig6_trace.png'); plt.close(fig)
# Architecture diagram
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=c, ec='k', lw=0.8)); ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=7)
box(0.1, 2.6, 2.4, 1.4, 'Batteryless sensor node i\n(solar harvester + supercap)\nprocess x_i(t)', '#e8f1fb')
box(0.1, 0.3, 2.4, 1.9, 'On-node logic\n1) Gateway-replica AR(1)/Kalman\n    predictor -> VoI score\n2) Energy-reserve shield\n3) Shared actor pi_theta(a|o_i)\n    a = {idle, (SF, P_tx)}', '#fdf1e0')
box(3.4, 1.4, 2.4, 1.4, 'LoRa uplink channel\nimperfect SF orthogonality\ncapture effect, Rayleigh fading\n1% duty cycle', '#eef7ea')
box(6.6, 2.6, 3.2, 1.4, 'Gateway / network server\nremote estimator x_hat_i(t)\nACK in RX1', '#e8f1fb')
box(6.6, 0.3, 3.2, 1.9, 'Centralized training (CTDE)\nglobal critic V_phi(o_i, g)\nLagrangian dual update of lambda\nLyapunov virtual energy queues Z_i\n-> periodic actor broadcast', '#f6e8f5')
for (x1, y1, x2, y2) in [(1.3, 2.6, 1.3, 2.2), (2.5, 1.2, 3.4, 2.0), (5.8, 2.1, 6.6, 3.2), (8.2, 2.6, 8.2, 2.2), (6.6, 1.0, 2.5, 0.8)]:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=0.9))
ax.text(4.5, 0.65, 'policy parameters (offline / periodic)', fontsize=6.5, ha='center')
fig.tight_layout(); fig.savefig('fig/fig1_architecture.png'); plt.close(fig)
print('figs ok')
