"""Small fully connected network trained with NumPy only (ReLU hidden, softmax output, Adam)."""
import numpy as np


class MLP:
    def __init__(self, dims, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / din), (dout, din)) for din, dout in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(dout) for dout in dims[1:]]

    def forward(self, x, keep=False):
        acts = [x]
        a = x
        for li, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W.T + b
            a = np.maximum(z, 0) if li < len(self.W) - 1 else z
            acts.append(a)
        return (a, acts) if keep else a

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def fit(self, X, y, epochs=300, batch=128, lr=3e-3, wd=1e-4, class_weight=None, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        n_cls = self.W[-1].shape[0]
        cw = np.ones(n_cls) if class_weight is None else np.asarray(class_weight, float)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        for ep in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                idx = order[s:s + batch]
                xb, yb = X[idx], y[idx]
                out, acts = self.forward(xb, keep=True)
                p = np.exp(out - out.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                wgt = cw[yb] / cw[yb].sum()
                g = p.copy()
                g[np.arange(len(yb)), yb] -= 1.0
                g *= wgt[:, None]
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for li in range(len(self.W) - 1, -1, -1):
                    gW[li] = g.T @ acts[li] + wd * self.W[li]
                    gb[li] = g.sum(axis=0)
                    if li > 0:
                        g = (g @ self.W[li]) * (acts[li] > 0)
                t += 1
                for i, (prm, gr) in enumerate(zip(params, gW + gb)):
                    m[i] = b1 * m[i] + (1 - b1) * gr
                    v[i] = b2 * v[i] + (1 - b2) * gr * gr
                    prm -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)
            if verbose and (ep + 1) % 50 == 0:
                print(f"  epoch {ep + 1}: train acc {np.mean(self.predict(X) == y):.3f}")
        return self
