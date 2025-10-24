# CosmoControlOpt — Astronomical Pointing + Control Gain Optimization
**Fun, higher-accuracy, visualization-rich** project that mixes **astronomy** (star tracker) with **control optimization**.
CosmoControlOpt2 is more responsive than CosmoControlOpt
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

<img width="1020" height="544" alt="response_best" src="https://github.com/user-attachments/assets/c17f2075-de25-4c84-b853-c2fa9190a0a2" />
<img width="1020" height="544" alt="torque_best" src="https://github.com/user-attachments/assets/a51f5945-bacd-4191-af61-26af276edde5" />
<img width="960" height="768" alt="landscape_heatmap" src="https://github.com/user-attachments/assets/7f12fba0-0a4b-4b64-bf7c-e28f0e44cfb4" />
<img width="951" height="714" alt="violin_J" src="https://github.com/user-attachments/assets/fba25186-7500-4df3-8133-0fc4d332d534" />
<img width="1020" height="544" alt="torque_best" src="https://github.com/user-attachments/assets/dccaeb30-e41a-4710-89ae-1d101a4934c1" />
<img width="1020" height="544" alt="response_best" src="https://github.com/user-attachments/assets/1c7d6ab4-741f-4646-9419-9a87de083f20" />
<img width="951" height="714" alt="pareto" src="https://github.com/user-attachments/assets/ca1acb4e-c7d0-467e-a6c8-55ec66beaf5e" />
