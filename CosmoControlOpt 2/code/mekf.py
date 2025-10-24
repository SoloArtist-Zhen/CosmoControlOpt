
import numpy as np
from .utils import quat_mul, quat_norm, quat_inv

def small_angle_from_quat(dq):
    dq = dq/np.linalg.norm(dq)
    return 2.0*dq[1:]

def quat_from_omega(w, dt):
    n = np.linalg.norm(w)
    if n < 1e-12:
        return np.array([1,0,0,0])
    axis = w/n; s = np.sin(n*dt/2.0)
    return np.array([np.cos(n*dt/2.0), *(axis*s)])

def quat_to_R(q):
    q = q/np.linalg.norm(q)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

class MEKFBiasOnly:
    def __init__(self, q0=None, b0=None, P0=None, Q=1e-7*np.eye(3), R=1e-5*np.eye(3)):
        self.q = np.array([1,0,0,0],dtype=float) if q0 is None else q0/np.linalg.norm(q0)
        self.b = np.zeros(3) if b0 is None else b0.copy()
        self.P = (1e-6*np.eye(3) if P0 is None else P0.copy())
        self.Q = Q
        self.R = R

    def predict(self, omega_meas, dt):
        dq = quat_from_omega(omega_meas - self.b, dt)
        self.q = quat_mul(self.q, dq); self.q = self.q/np.linalg.norm(self.q)
        self.P = self.P + self.Q*dt

    def update(self, q_meas):
        w,x,y,z = self.q
        qi = np.array([w,-x,-y,-z])
        dq = np.array([
            q_meas[0]*qi[0] - np.dot(q_meas[1:], qi[1:]),
            q_meas[0]*qi[1] + qi[0]*q_meas[1] + q_meas[2]*qi[3] - q_meas[3]*qi[2],
            q_meas[0]*qi[2] - qi[3]*q_meas[1] + qi[0]*q_meas[2] + q_meas[1]*qi[3],
            q_meas[0]*qi[3] + qi[2]*q_meas[1] - qi[1]*q_meas[2] + qi[0]*q_meas[3]
        ])
        delta = small_angle_from_quat(dq)
        H = np.eye(3)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        db = K @ delta
        self.b += db
        self.P = (np.eye(3) - K@H) @ self.P
        corr = np.array([1.0, *(0.5*db)])
        self.q = quat_mul(corr, self.q)
        self.q = self.q/np.linalg.norm(self.q)
        return delta, db

def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w = (R[2,1]-R[1,2])/s; x = 0.25*s; y = (R[0,1]+R[1,0])/s; z = (R[0,2]+R[2,0])/s
        elif i == 1:
            s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w = (R[0,2]-R[2,0])/s; x = (R[0,1]+R[1,0])/s; y = 0.25*s; z = (R[1,2]+R[2,1])/s
        else:
            s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w = (R[1,0]-R[0,1])/s; x = (R[0,2]+R[2,0])/s; y = (R[1,2]+R[2,1])/s; z = 0.25*s
    q = np.array([w,x,y,z])
    return q/np.linalg.norm(q)
