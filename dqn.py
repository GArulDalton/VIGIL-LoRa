import numpy as np, pickle, time
from env import LoRaEHEnv, N_ACT
from nn import MLP
def train_dqn(steps=72000, N=200, seed=0):
    rng = np.random.default_rng(seed)
    env = LoRaEHEnv(N=N, R=3000, seed=seed+100)
    q = MLP([12, 64, 64, N_ACT], rng, out_scale=0.1); tq = MLP([12, 64, 64, N_ACT], rng); tq.copy_from(q)
    cap = 400000; O = np.zeros((cap, 12), np.float32); O2 = np.zeros_like(O); A = np.zeros(cap, int)
    Rw = np.zeros(cap, np.float32); M2 = np.zeros((cap, N_ACT), bool); ptr = 0; full = False
    o = env.obs(); hist = []; nup = 0
    for t in range(steps):
        if t > 0 and t % 3600 == 0:
            env = LoRaEHEnv(N=int(rng.choice([100, 200, 300, 400])), R=3000, seed=seed+100+t, start_hour=float(rng.uniform(0, 24)))
            N = env.N; o = env.obs()
        eps = max(0.05, 1 - t/(0.6*steps))
        mk = env.mask(); qv = np.where(mk, q.forward(o), -1e9)
        a = qv.argmax(1)
        rnd = rng.random(N) < eps
        for i in np.where(rnd)[0]:
            a[i] = rng.choice(np.where(mk[i])[0])
        e2, out, info = env.step(a)
        r = -e2/5.0 - 1.0*out
        o2 = env.obs(); mk2 = env.mask()
        idx = (ptr + np.arange(N)) % cap
        O[idx], A[idx], Rw[idx], O2[idx], M2[idx] = o, a, r, o2, mk2
        ptr = (ptr + N) % cap; full = full or ptr < N
        o = o2
        if t % 2 == 0 and t > 50:
            n = cap if full else ptr
            for _ in range(2):
                j = rng.integers(0, n, 1024)
                tgt = Rw[j] + 0.99*np.where(M2[j], tq.forward(O2[j]), -1e9).max(1)
                qa = q.forward(O[j], cache=True)
                g = np.zeros_like(qa); g[np.arange(1024), A[j]] = (qa[np.arange(1024), A[j]] - tgt)/1024
                g = np.clip(g, -1/1024, 1/1024)  # Huber
                q.adam(q.backward(g.astype(np.float32)), 5e-4, clip=10.0); nup += 1
                if nup % 250 == 0: tq.copy_from(q)
        if t % 360 == 0:
            hist.append(dict(t=t, nmse=float(e2.mean()), eps=eps)); print('dqn', t, hist[-1], flush=True)
    pickle.dump(dict(q=q, hist=hist), open('models/dqn.pkl', 'wb'))
if __name__ == '__main__':
    t0 = time.time(); train_dqn(); print('done dqn', time.time()-t0)
