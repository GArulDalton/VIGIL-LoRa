import numpy as np, pickle
def pol_dqn(path='models/dqn.pkl'):
    q = pickle.load(open(path, 'rb'))['q']
    def f(env, st):
        return np.where(env.mask(), q.forward(env.obs()), -1e9).argmax(1)
    return f
