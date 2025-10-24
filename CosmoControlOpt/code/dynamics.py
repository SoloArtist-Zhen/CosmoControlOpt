
import numpy as np
from .utils import integrate_attitude

class Spacecraft:
    def __init__(self, J=np.diag([0.08,0.07,0.05]), umax=0.2):
        self.J = J
        self.umax = umax

    def step(self, q, w, u, dt, dist=np.zeros(3)):
        u = np.clip(u, -self.umax, self.umax)
        J = self.J
        Jw = J@w
        wdot = np.linalg.inv(J) @ (u - np.cross(w, Jw) + dist)
        w_new = w + wdot*dt
        q_new = integrate_attitude(q, w, dt)
        return q_new, w_new, u
