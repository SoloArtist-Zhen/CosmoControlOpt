
import numpy as np

def clip_bounds(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def grid_search(fun, lo, hi, n_per_dim=8):
    grids = [np.linspace(lo[i], hi[i], n_per_dim) for i in range(len(lo))]
    best = None; bestx = None; hist = []
    for a in grids[0]:
        for b in grids[1]:
            x = np.array([a,b])
            f = fun(x)
            hist.append(f)
            if best is None or f < best:
                best, bestx = f, x.copy()
    return bestx, best, np.array(hist)

def random_search(fun, lo, hi, iters=60, seed=0):
    rng = np.random.RandomState(seed)
    best = None; bestx = None; hist = []
    for _ in range(iters):
        x = lo + rng.rand(len(lo))*(hi-lo)
        f = fun(x)
        hist.append(f)
        if best is None or f < best:
            best, bestx = f, x.copy()
    return bestx, best, np.array(hist)

def nelder_mead(fun, x0, step=0.2, iters=80, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    n = len(x0)
    simplex = [x0]
    for i in range(n):
        e = np.zeros(n); e[i]=1.0
        simplex.append(x0 + step*e)
    simplex = np.array(simplex)
    hist = []
    for _ in range(iters):
        vals = np.array([fun(x) for x in simplex]); hist.append(np.min(vals))
        idx = np.argsort(vals); simplex = simplex[idx]; vals = vals[idx]
        x0m = np.mean(simplex[:-1], axis=0)
        xr = x0m + alpha*(x0m - simplex[-1])
        fr = fun(xr)
        if vals[0] <= fr < vals[-2]:
            simplex[-1] = xr; continue
        if fr < vals[0]:
            xe = x0m + gamma*(xr - x0m)
            fe = fun(xe)
            simplex[-1] = xe if fe < fr else xr; continue
        xc = x0m + rho*(simplex[-1] - x0m)
        fc = fun(xc)
        if fc < vals[-1]:
            simplex[-1] = xc; continue
        best = simplex[0]
        for i in range(1, len(simplex)):
            simplex[i] = best + sigma*(simplex[i]-best)
    return simplex[0], fun(simplex[0]), np.array(hist)

def simulated_annealing(fun, lo, hi, iters=120, T0=1.0, decay=0.98, seed=0):
    rng = np.random.RandomState(seed)
    x = lo + rng.rand(len(lo))*(hi-lo)
    fx = fun(x)
    hist = [fx]
    T = T0
    for _ in range(iters):
        step = (hi-lo)*0.15*rng.randn(len(lo))
        xn = clip_bounds(x + step, lo, hi)
        fn = fun(xn)
        if fn < fx or rng.rand() < np.exp((fx-fn)/max(T,1e-8)):
            x, fx = xn, fn
        hist.append(fx)
        T *= decay
    return x, fx, np.array(hist)

def pso(fun, lo, hi, iters=40, swarm=16, w=0.7, c1=1.4, c2=1.4, seed=0):
    rng = np.random.RandomState(seed)
    dim = len(lo)
    x = lo + rng.rand(swarm, dim)*(hi-lo)
    v = rng.randn(swarm, dim)*0.1
    p = x.copy()
    fp = np.array([fun(xx) for xx in x])
    g_idx = np.argmin(fp); g = x[g_idx].copy(); fg = fp[g_idx]
    hist = [fg]
    for _ in range(iters):
        r1, r2 = rng.rand(swarm, dim), rng.rand(swarm, dim)
        v = w*v + c1*r1*(p-x) + c2*r2*(g-x)
        x = clip_bounds(x+v, lo, hi)
        f = np.array([fun(xx) for xx in x])
        better = f < fp
        p[better] = x[better]; fp[better]=f[better]
        if f.min() < fg:
            idx = np.argmin(f); g = x[idx].copy(); fg = f[idx]
        hist.append(fg)
    return g, fg, np.array(hist)
