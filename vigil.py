"""VIGIL-LoRa: constrained parameter-shared MAPPO (CTDE) with VoI observations/rewards and Lyapunov shaping."""
import numpy as np, time, json, pickle, sys
from env import LoRaEHEnv, N_ACT, E_TX_TABLE, AIR_TABLE
from nn import MLP, masked_softmax

class Cfg:
    N=200; T=360; iters=120; gamma=0.995; lam=0.95; clip=0.2; epochs=4; mb=4096
    lr_a=3e-4; lr_c=1e-3; ent0=0.02; ent1=0.002
    use_voi=True; use_lyap=True; constrained=True; central=True; ideal=False
    delta=0.02; lam_lr=0.2; lam0=0.5; lam_max=10.0; kappa=0.1; seed=0; shield=True; fixed_lambda=0.0; zeta=0.03; rebuild=10

def local_obs(env, cfg):
    o = env.obs()
    if not cfg.use_voi: o[:, 0] = 0.0; o[:, 10] = 0.0
    if not cfg.use_lyap: o[:, 7] = 0.0
    return o

def reward(env, info, cfg, lam):
    if cfg.use_voi: r = -info['e2']/5.0
    else: r = -np.minimum(env.aoi, 100)/20.0
    if cfg.use_lyap:
        r = r - cfg.kappa*np.minimum(env.Z/(0.2*env.E_max), 5.0)*(info['etx']/0.05)
    r = r - cfg.zeta*info['tx']*(AIR_TABLE[np.where(info['tx'], 1+3*info['sfi'], 0)]/AIR_TABLE[1])**0.5
    c = info['outage'].astype(np.float32)
    return (r - lam*c)/(1.0+lam), c

def train(cfg, tag, log=True):
    rng = np.random.default_rng(cfg.seed)
    env = LoRaEHEnv(N=cfg.N, R=3000, seed=cfg.seed+100, ideal_orth=cfg.ideal, shield=cfg.shield)
    obs_d, g_d = 12, 8
    actor = MLP([obs_d, 64, 64, N_ACT], rng)
    actor.b[-1][0] = 4.0
    critic = MLP([obs_d + (g_d if cfg.central else 0), 64, 64, 1], rng, out_scale=1.0)
    lam = cfg.lam0 if cfg.constrained else cfg.fixed_lambda
    hist = []
    for it in range(cfg.iters):
        if it > 0 and it % cfg.rebuild == 0:   # topology / density randomisation
            env = LoRaEHEnv(N=int(rng.choice([100, 200, 300, 400])), R=3000, seed=cfg.seed+100+it,
                            ideal_orth=cfg.ideal, shield=cfg.shield, start_hour=float(rng.uniform(0, 24)))
        T, N = cfg.T, env.N
        O = np.zeros((T, N, obs_d), np.float32); GO = np.zeros((T, N, g_d), np.float32)
        Mk = np.zeros((T, N, N_ACT), bool); A = np.zeros((T, N), int); LP = np.zeros((T, N), np.float32)
        R = np.zeros((T, N), np.float32); C = np.zeros((T, N), np.float32); V = np.zeros((T+1, N), np.float32)
        e2s = []; txs = []
        for t in range(T):
            o = local_obs(env, cfg); g = env.gobs(); mk = env.mask()
            p = masked_softmax(actor.forward(o), mk)
            a = (p.cumsum(1) > rng.random((N, 1))).argmax(1)
            a = np.where(mk[np.arange(N), a], a, 0)
            ci = np.concatenate([o, g], 1) if cfg.central else o
            V[t] = critic.forward(ci)[:, 0]
            O[t], GO[t], Mk[t], A[t] = o, g, mk, a
            LP[t] = np.log(p[np.arange(N), a] + 1e-12)
            e2, out, info = env.step(a)
            R[t], C[t] = reward(env, info, cfg, lam)
            e2s.append(e2.mean()); txs.append(info['tx'].mean())
        o = local_obs(env, cfg); g = env.gobs()
        V[T] = critic.forward(np.concatenate([o, g], 1) if cfg.central else o)[:, 0]
        adv = np.zeros((T, N), np.float32); last = 0
        for t in reversed(range(T)):
            d = R[t] + cfg.gamma*V[t+1] - V[t]
            last = d + cfg.gamma*cfg.lam*last; adv[t] = last
        ret = adv + V[:T]
        # flatten
        fo = O.reshape(-1, obs_d); fg = GO.reshape(-1, g_d); fm = Mk.reshape(-1, N_ACT)
        fa = A.reshape(-1); flp = LP.reshape(-1); fadv = adv.reshape(-1); fret = ret.reshape(-1)
        fadv = (fadv - fadv.mean())/(fadv.std()+1e-8)
        ent_c = cfg.ent0 + (cfg.ent1-cfg.ent0)*it/max(cfg.iters-1, 1)
        B = len(fa)
        for ep in range(cfg.epochs):
            perm = rng.permutation(B)
            for s in range(0, B, cfg.mb):
                j = perm[s:s+cfg.mb]; b = len(j)
                z = actor.forward(fo[j], cache=True)
                p = masked_softmax(z, fm[j])
                lp = np.log(p[np.arange(b), fa[j]] + 1e-12)
                ratio = np.exp(lp - flp[j]); Aj = fadv[j]
                s1 = ratio*Aj; s2 = np.clip(ratio, 1-cfg.clip, 1+cfg.clip)*Aj
                act = (s1 <= s2).astype(np.float32)
                dlp = -act*ratio*Aj/b
                oh = np.zeros_like(p); oh[np.arange(b), fa[j]] = 1
                gz = dlp[:, None]*(oh - p)
                logp = np.where(p > 0, np.log(p + 1e-12), 0)
                H = -(p*logp).sum(1)
                gz += ent_c*(p*(logp + H[:, None]))/b    # gradient of -ent_c*H
                gz = np.where(fm[j], gz, 0).astype(np.float32)
                actor.adam(actor.backward(gz), cfg.lr_a)
                ci = np.concatenate([fo[j], fg[j]], 1) if cfg.central else fo[j]
                v = critic.forward(ci, cache=True)[:, 0]
                gv = (2*(v - fret[j])/b)[:, None].astype(np.float32)*0.5
                critic.adam(critic.backward(gv), cfg.lr_c)
        out_rate = C.mean()
        if cfg.constrained:
            lam = float(np.clip(lam + cfg.lam_lr*(out_rate - cfg.delta), 0.0, cfg.lam_max))
        hist.append(dict(it=it, nmse=float(np.mean(e2s)), outage=float(out_rate), lam=float(lam),
                         tx=float(np.mean(txs)), ret=float(R.mean())))
        if log and it % 10 == 0:
            print(tag, hist[-1], flush=True)
    with open(f'models/{tag}.pkl', 'wb') as f:
        pickle.dump(dict(actor=actor, critic=critic, cfg={k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}, hist=hist, lam=lam), f)
    return actor, hist

VARIANTS = {
    'vigil': {},
    'no_voi': dict(use_voi=False),
    'no_lyap': dict(use_lyap=False),
    'no_constr': dict(constrained=False, fixed_lambda=0.0),
    'no_central': dict(central=False),
    'ideal_orth': dict(ideal=True),
    'no_shield': dict(shield=False),
}
if __name__ == '__main__':
    tag = sys.argv[1]; iters = int(sys.argv[2]) if len(sys.argv) > 2 else Cfg.iters
    c = Cfg(); c.iters = iters
    for k, v in VARIANTS[tag].items(): setattr(c, k, v)
    t0 = time.time(); train(c, tag); print('done', tag, time.time()-t0, flush=True)
