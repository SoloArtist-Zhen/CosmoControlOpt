
import numpy as np, matplotlib.pyplot as plt, json, argparse
from pathlib import Path
from .utils import quat_from_axis_angle, quat_to_R, R_to_quat, quat_mul, quat_inv, ang_between_R
from .dynamics import Spacecraft
from .controller import pd_torque
from .startracker import StarTracker
from .mekf import MEKFBiasOnly, R_to_quat as R2Q
from .objective import evaluate_metrics, objective
from .optimizers import grid_search, random_search, nelder_mead, simulated_annealing, pso

OUT = Path(__file__).resolve().parents[1]/"outputs"
OUT.mkdir(exist_ok=True, parents=True)

def simulate_once(Kp, Kd, seed=0, T=20.0, dt=0.02, umax=0.2, fov_deg=20, noise=6e-4):
    np.random.seed(seed)
    sc = Spacecraft(umax=umax)
    st = StarTracker(n_stars_catalog=1800, fov_deg=fov_deg, dir_noise_std=noise)
    axis = np.random.randn(3); axis/=np.linalg.norm(axis)
    q = quat_from_axis_angle(axis, np.deg2rad(15.0))
    w = np.deg2rad(np.array([0.2,-0.3,0.15]))
    qd = np.array([1,0,0,0])
    mekf = MEKFBiasOnly()
    steps = int(T/dt)
    err_hist = []; u_hist = []; w_hist = []; t_hist = []
    sat_count = 0
    for k in range(steps):
        t = k*dt
        Rtrue = quat_to_R(q)
        q_meas = st.observe(Rtrue)
        if q_meas is None:
            q_meas = R2Q(Rtrue)
        mekf.predict(w + 0.002*np.random.randn(3), dt)
        mekf.update(q_meas)
        u, q_e = pd_torque(qd, mekf.q, w, Kp, Kd)
        d = 0.01*np.random.randn(3)
        q, w, u_sat = sc.step(q, w, u, dt, dist=d)
        err = ang_between_R(quat_to_R(mekf.q), quat_to_R(qd))
        err_hist.append(err); u_hist.append(u_sat); w_hist.append(w); t_hist.append(t)
        if np.any(np.abs(u_sat) >= (umax-1e-6)): sat_count += 1
    err_hist = np.array(err_hist); u_hist = np.array(u_hist); w_hist=np.array(w_hist); t_hist=np.array(t_hist)
    settle, overshoot, final_err, effort, sat_ratio = evaluate_metrics(t_hist, err_hist, w_hist, u_hist, sat_count/len(t_hist))
    J = objective(settle, overshoot, final_err, effort, sat_ratio)
    return J, {"settle":settle, "overshoot":overshoot, "final_err":final_err, "effort":effort, "sat_ratio":sat_ratio, "t":t_hist, "err":err_hist, "u":u_hist, "w":w_hist}

def run_optimizer(name, fun, lo, hi, heavy=False, seed=0):
    if name=="grid":
        n = 12 if heavy else 7
        return grid_search(fun, lo, hi, n_per_dim=n)
    if name=="random":
        it = 180 if heavy else 60
        return random_search(fun, lo, hi, iters=it, seed=seed)
    if name=="nm":
        it = 180 if heavy else 80
        return nelder_mead(fun, (lo+hi)/2, step=0.25*(hi-lo), iters=it)
    if name=="sa":
        it = 220 if heavy else 120
        return simulated_annealing(fun, lo, hi, iters=it, T0=1.0, decay=0.985, seed=seed)
    if name=="pso":
        it = 120 if heavy else 40
        swarm = 24 if heavy else 16
        return pso(fun, lo, hi, iters=it, swarm=swarm, seed=seed)
    raise ValueError("unknown optimizer")

def plot_landscape(fun, lo, hi, N=60):
    xs = np.linspace(lo[0], hi[0], N)
    ys = np.linspace(lo[1], hi[1], N)
    Z = np.zeros((N,N))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            Z[j,i] = fun(np.array([x,y]))
    plt.figure(figsize=(6,4.8))
    plt.imshow(Z, origin="lower", extent=[lo[0],hi[0],lo[1],hi[1]], aspect="auto")
    plt.colorbar(label="Objective J")
    plt.xlabel("Kp"); plt.ylabel("Kd"); plt.title("Objective Landscape (heatmap)")
    plt.tight_layout(); plt.savefig(OUT/"landscape_heatmap.png", dpi=160); plt.close()

def plot_convergence(histories, names):
    plt.figure(figsize=(6,3.2))
    for h in histories:
        plt.plot(h, alpha=0.9)
    plt.xlabel("Iteration"); plt.ylabel("Best-so-far J"); plt.title("Optimizer Convergence")
    plt.legend(names)
    plt.grid(True); plt.tight_layout(); plt.savefig(OUT/"convergence.png", dpi=170); plt.close()

def plot_response(best_runs):
    plt.figure(figsize=(6,3.2))
    for name, dat in best_runs.items():
        plt.plot(dat["t"], np.rad2deg(np.abs(dat["err"])), label=name)
    plt.xlabel("Time (s)"); plt.ylabel("|Attitude error| (deg)"); plt.title("Time Response of Best Designs")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"response_best.png", dpi=170); plt.close()

def plot_torque(best_runs):
    plt.figure(figsize=(6,3.2))
    for name, dat in best_runs.items():
        tau = np.linalg.norm(dat["u"], axis=1)
        plt.plot(dat["t"], tau, label=name)
    plt.xlabel("Time (s)"); plt.ylabel("||u||"); plt.title("Control Effort of Best Designs")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"torque_best.png", dpi=170); plt.close()

def plot_pareto(scores):
    plt.figure(figsize=(5.6,4.2))
    for name, s in scores.items():
        plt.scatter(s["settle"], s["effort"], label=name, s=60)
        plt.annotate(name, (s["settle"], s["effort"]), xytext=(4,4), textcoords="offset points", fontsize=8)
    plt.xlabel("Settle time (s)"); plt.ylabel("Effort ∫||u||dt"); plt.title("Pareto (Settle vs Effort)")
    plt.grid(True); plt.tight_layout(); plt.savefig(OUT/"pareto.png", dpi=170); plt.close()

def plot_violin(dists, label):
    plt.figure(figsize=(5.6,4.2))
    plt.violinplot(dists, showmeans=True, showextrema=True, showmedians=True)
    plt.title(label); plt.xlabel("Algorithm index"); plt.ylabel("Objective J")
    plt.tight_layout(); plt.savefig(OUT/"violin_J.png", dpi=170); plt.close()

def main():
    import time
    t0 = time.time()
    # Bounds for Kp, Kd
    lo = np.array([0.1, 0.02])
    hi = np.array([3.5, 0.6])

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy", action="store_true")
    args = ap.parse_args()

    seeds_train = [0,1] if not args.heavy else [0,1,2]
    def J_of_x(x):
        vals = []
        for sd in seeds_train:
            J, _ = simulate_once(Kp=x[0], Kd=x[1], seed=sd, T=20.0 if not args.heavy else 25.0)
            vals.append(J)
        return float(np.mean(vals))

    plot_landscape(J_of_x, lo, hi, N=40 if not args.heavy else 55)

    optimizers = ["grid","random","nm","sa","pso"]
    histories = []
    bests = {}
    for i,name in enumerate(optimizers):
        xbest, fbest, hist = run_optimizer(name, J_of_x, lo, hi, heavy=args.heavy, seed=42+i)
        histories.append(hist)
        J, detail = simulate_once(Kp=xbest[0], Kd=xbest[1], seed=123, T=18.0 if not args.heavy else 22.0)
        bests[name] = detail
        bests[name]["J"] = J
        bests[name]["Kp"] = float(xbest[0]); bests[name]["Kd"] = float(xbest[1])

    plot_convergence(histories, optimizers)
    plot_response(bests)
    plot_torque(bests)
    pareto = {k: {"settle":v["settle"], "effort":v["effort"]} for k,v in bests.items()}
    plot_pareto(pareto)

    dists = []
    for i,name in enumerate(optimizers):
        xs = []
        for r in range(3):
            xb, fb, _ = run_optimizer(name, J_of_x, lo, hi, heavy=False, seed=100+i*10+r)
            xs.append(J_of_x(xb))
        dists.append(xs)
    plot_violin(dists, "Distribution of Best-found J")

    (OUT/"benchmarks.json").write_text(json.dumps({
        "best": {k: {"J":v["J"], "Kp":v["Kp"], "Kd":v["Kd"],
                    "settle":v["settle"], "overshoot":v["overshoot"],
                    "final_err":v["final_err"], "effort":v["effort"], "sat_ratio":v["sat_ratio"]} for k,v in bests.items()},
        "elapsed_s": round(time.time()-t0,2)
    }, indent=2))

if __name__ == "__main__":
    main()
