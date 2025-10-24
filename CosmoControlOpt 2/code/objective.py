
import numpy as np

def evaluate_metrics(t_hist, err_hist, w_hist, u_hist, sat_ratio):
    deg = np.rad2deg(np.abs(err_hist))
    settle = None
    for i in range(len(deg)-1, -1, -1):
        if np.max(deg[i:]) <= 1.0:
            settle = t_hist[i]; break
    if settle is None:
        settle = t_hist[-1] + 5.0
    overshoot = float(np.max(deg) - deg[-1])
    final_err = float(deg[-1])
    effort = float(np.trapz(np.linalg.norm(u_hist, axis=1), t_hist))
    return settle, overshoot, final_err, effort, float(sat_ratio)

def objective(settle, overshoot, final_err, effort, sat_ratio, w=(0.5,0.1,0.2,0.15,0.05)):
    return w[0]*settle + w[1]*overshoot + w[2]*final_err + w[3]*effort + w[4]*sat_ratio
