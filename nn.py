import numpy as np
class MLP:
    def __init__(self, sizes, rng, out_scale=0.01):
        self.W, self.b = [], []
        for i in range(len(sizes)-1):
            s = np.sqrt(2.0/sizes[i]) if i < len(sizes)-2 else out_scale
            self.W.append((rng.normal(0, 1, (sizes[i], sizes[i+1]))*s).astype(np.float32))
            self.b.append(np.zeros(sizes[i+1], np.float32))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]; self.v = [np.zeros_like(p) for p in self.params]
        self.k = 0
    def forward(self, x, cache=False):
        hs = [x]; h = x
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z) if i < len(self.W)-1 else z
            hs.append(h)
        if cache: self.hs = hs
        return h
    def backward(self, g):
        gW, gb = [None]*len(self.W), [None]*len(self.W)
        for i in reversed(range(len(self.W))):
            h_in = self.hs[i]
            gW[i] = h_in.T @ g; gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T)*(1 - self.hs[i]**2)
        return gW + gb
    def adam(self, grads, lr, clip=0.5, b1=0.9, b2=0.999, eps=1e-8):
        n = np.sqrt(sum((g**2).sum() for g in grads))
        if n > clip: grads = [g*clip/n for g in grads]
        self.k += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1; m += (1-b1)*g; v *= b2; v += (1-b2)*g*g
            mh = m/(1-b1**self.k); vh = v/(1-b2**self.k)
            p -= (lr*mh/(np.sqrt(vh)+eps)).astype(np.float32)
    def copy_from(self, o):
        for p, q in zip(self.params, o.params): p[...] = q
    def n_params(self):
        return sum(p.size for p in self.params)

def masked_softmax(z, mask):
    z = np.where(mask, z, -1e9); z = z - z.max(1, keepdims=True)
    e = np.exp(z)*mask; p = e/e.sum(1, keepdims=True)
    return p
