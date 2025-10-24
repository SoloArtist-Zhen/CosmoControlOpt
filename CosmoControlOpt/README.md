# CosmoControlOpt — Astronomical Pointing + Control Gain Optimization
**Fun, higher-accuracy, visualization-rich** project that mixes **astronomy** (star tracker) with **control optimization**.

### What it does
- Simulates a rigid spacecraft doing a **slew to point at a star**.
- Generates synthetic **star tracker** measurements (Wahba SVD) and fuses with gyro using a **MEKF-lite**.
- Uses a quaternion **PD controller** with reaction-wheel torque saturation.
- Tunes controller gains **(Kp, Kd)** via **five optimizers** and compares them:
  1) Grid Search
  2) Random Search
  3) Nelder–Mead
  4) Simulated Annealing
  5) Particle Swarm Optimization (PSO)
- Produces **advanced plots**: objective landscape heatmap, convergence curves, time responses, torque profiles, Pareto chart, violin distributions.

### Why it's useful
- Gives a **reproducible baseline** for star-tracker–inertial **fusion + control** problems.
- Lets you **compare optimizers** on a physical control task with noisy sensing & saturations.
- Ready to extend to event camera, multi-axis inertia, flexible appendages, delay, etc.

### Run
```bash
python code/main.py            # quick run (short iterations)
python code/main.py --heavy    # more evaluations for clearer differences
```
Outputs go to `outputs/`.

No external deps beyond `numpy`, `matplotlib`. This repo does **not** require seaborn or others.
