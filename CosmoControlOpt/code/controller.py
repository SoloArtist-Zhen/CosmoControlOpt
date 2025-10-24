
import numpy as np
from .utils import quat_inv

def quat_error(qd, q):
    qi = quat_inv(qd)
    w1,x1,y1,z1 = qi
    w2,x2,y2,z2 = q
    q_e = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    if q_e[0] < 0: q_e = -q_e
    return q_e

def pd_torque(qd, q, w, Kp, Kd):
    q_e = quat_error(qd, q)
    u = -Kp*q_e[1:] - Kd*w
    return u, q_e
