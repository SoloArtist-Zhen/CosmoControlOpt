# Whitepaper (Short): Star-Tracker–Inertial Fusion with Gain Optimization
**Problem.** Slew a rigid body to a target celestial direction under sensing noise and actuator saturation. Estimate attitude from a star tracker (Wahba SVD) fused with gyro via MEKF-lite, and tune PD gains (Kp, Kd).

**Dynamics.** \(\dot q = \tfrac12 \Omega(\omega) q\), \(J\dot\omega + \omega\times J\omega = u + d\). Reaction wheel torque saturates at \(u_{max}\).

**Sensing.** Inertial star directions \(s_i\). Camera-frame observations with direction noise. Wahba solves \(\min_R \sum w_i \| r_i - R s_i \|^2 \), SVD gives \(\hat R\). MEKF-lite estimates bias \(b_g\) and corrects attitude with small-angle innovation.

**Control.** Quaternion error \(q_e = q_d^{-1} \otimes q\). PD torque \(u = -K_p e_{vec} - K_d \omega\).

**Optimization.** Compare 5 algorithms over \(K_p, K_d\) in bounds. Objective combines settle time, final error, control effort, overshoot, and saturation ratio. We report convergence, best-found parameters, and Pareto plot.
