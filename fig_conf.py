import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
R = json.load(open('results/res_conformal.json')); K = [1.0, 1.5, 2.0, 2.5]
fig, ax = plt.subplots(1, 2, figsize=(6.85, 2.6))
for name, c, m, ls in [('model', 'C0', 'o', '--'), ('conformal', 'C3', 's', '-')]:
    lab = 'Model-based shield (μ = 1.2)' if name == 'model' else 'Conformal shield (μ̂)'
    y = [R[f'{k}|{name}']['cond_night_outage'][0] for k in K]; e = [R[f'{k}|{name}']['cond_night_outage'][1] for k in K]
    ax[0].errorbar(K, y, yerr=e, color=c, marker=m, ls=ls, capsize=2, label=lab)
    y = [R[f'{k}|{name}']['nmse'][0] for k in K]; e = [R[f'{k}|{name}']['nmse'][1] for k in K]
    ax[1].errorbar(K, y, yerr=e, color=c, marker=m, ls=ls, capsize=2, label=lab)
ax[0].axhline(0.05, color='k', lw=0.8, ls=':'); ax[0].text(1.02, 0.065, 'target ε = 0.05', fontsize=7)
ax[0].set_xlabel('Maximum drain mismatch κ_max'); ax[0].set_ylabel('Night outage of shield-qualified nodes'); ax[0].legend(fontsize=7)
ax[1].set_xlabel('Maximum drain mismatch κ_max'); ax[1].set_ylabel('NMSE'); ax[1].legend(fontsize=7)
for a in ax: a.set_xticks(K)
fig.tight_layout(); fig.savefig('fig/fig10_conformal.png', dpi=600)
print('ok')
