
import numpy as np, matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/servers
import matplotlib.pyplot as plt, json, argparse, time, sys
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

def simulate_once(Kp, Kd, seed=0, T=12.0, dt=0.03, umax=0.2, fov_deg=20, noise=6e-4):
    np.random.seed(seed)
    sc = Spacecraft(umax=umax)
    st = StarTracker(n_stars_catalog=1600, fov_deg=fov_deg, dir_noise_std=noise)
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
        n = 8 if heavy else 5
        print(f"[opt] grid_search n_per_dim={n}"); sys.stdout.flush()
        return grid_search(fun, lo, hi, n_per_dim=n)
    if name=="random":
        it = 160 if heavy else 40
        print(f"[opt] random_search iters={it} seed={seed}"); sys.stdout.flush()
        return random_search(fun, lo, hi, iters=it, seed=seed)
    if name=="nm":
        it = 160 if heavy else 60
        print(f"[opt] nelder_mead iters={it}"); sys.stdout.flush()
        return nelder_mead(fun, (lo+hi)/2, step=0.25*(hi-lo), iters=it)
    if name=="sa":
        it = 200 if heavy else 90
        print(f"[opt] simulated_annealing iters={it}"); sys.stdout.flush()
        return simulated_annealing(fun, lo, hi, iters=it, T0=1.0, decay=0.985, seed=seed)
    if name=="pso":
        it = 90 if heavy else 30
        swarm = 24 if heavy else 12
        print(f"[opt] pso iters={it} swarm={swarm}"); sys.stdout.flush()
        return pso(fun, lo, hi, iters=it, swarm=swarm, seed=seed)
    raise ValueError("unknown optimizer")

def plot_landscape(fun, lo, hi, N=18):
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
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy", action="store_true", help="more iterations for optimizers")
    ap.add_argument("--no-landscape", action="store_true", help="skip heatmap (fast)")
    ap.add_argument("--no-violin", action="store_true", help="skip violin plot (fast)")
    ap.add_argument("--optim", type=str, default="grid,random,nm,sa,pso", help="comma list of optimizers")
    ap.add_argument("--T", type=float, default=12.0, help="simulation horizon [s]")
    ap.add_argument("--dt", type=float, default=0.03, help="simulation step [s]")
    ap.add_argument("--landN", type=int, default=18, help="heatmap grid N")
    ap.add_argument("--seed", type=int, default=42, help="base seed for optimizers")
    args = ap.parse_args()

    lo = np.array([0.1, 0.02])
    hi = np.array([3.5, 0.6])

    seeds_train = [0] if not args.heavy else [0,1,2]
    def J_of_x(x):
        vals = []
        for sd in seeds_train:
            J, _ = simulate_once(Kp=x[0], Kd=x[1], seed=sd, T=args.T, dt=args.dt)
            vals.append(J)
        return float(np.mean(vals))

    if not args.no_landscape:
        print(f"[plot] landscape heatmap N={args.landN} (this is the slowest step)."); sys.stdout.flush()
        plot_landscape(J_of_x, lo, hi, N=args.landN)

    optimizers = [s.strip() for s in args.optim.split(",") if s.strip()]
    histories = []
    bests = {}
    for i,name in enumerate(optimizers):
        xbest, fbest, hist = run_optimizer(name, J_of_x, lo, hi, heavy=args.heavy, seed=args.seed+i)
        histories.append(hist)
        J, detail = simulate_once(Kp=xbest[0], Kd=xbest[1], seed=123, T=args.T, dt=args.dt)
        bests[name] = detail
        bests[name]["J"] = J
        bests[name]["Kp"] = float(xbest[0]); bests[name]["Kd"] = float(xbest[1])
        print(f"[done] {name}: J={J:.3f}, Kp={xbest[0]:.3f}, Kd={xbest[1]:.3f}")

    plot_convergence(histories, optimizers)
    plot_response(bests)
    plot_torque(bests)
    pareto = {k: {"settle":v["settle"], "effort":v["effort"]} for k,v in bests.items()}
    plot_pareto(pareto)

    if not args.no_violin:
        dists = []
        for i,name in enumerate(optimizers):
            xs = []
            for r in range(2 if not args.heavy else 4):
                xb, fb, _ = run_optimizer(name, J_of_x, lo, hi, heavy=False, seed=args.seed+100+i*10+r)
                xs.append(J_of_x(xb))
            dists.append(xs)
        plot_violin(dists, "Distribution of Best-found J")

    (OUT/"benchmarks.json").write_text(json.dumps({
        "best": {k: {"J":v["J"], "Kp":v["Kp"], "Kd":v["Kd"],
                    "settle":v["settle"], "overshoot":v["overshoot"],
                    "final_err":v["final_err"], "effort":v["effort"], "sat_ratio":v["sat_ratio"]} for k,v in bests.items()},
        "elapsed_s": round(time.time()-t0,2),
        "args": vars(args)
    }, indent=2))
    print("[ok] Finished. See outputs/*.png and benchmarks.json")

if __name__ == "__main__":
    main()
