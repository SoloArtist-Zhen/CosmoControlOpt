
import numpy as np

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_inv(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])/(np.dot(q,q)+1e-15)

def quat_norm(q):
    return q/np.linalg.norm(q)

def quat_from_axis_angle(axis, ang):
    a = np.asarray(axis, dtype=float); a = a/np.linalg.norm(a)
    s = np.sin(ang/2.0)
    return np.array([np.cos(ang/2.0), a[0]*s, a[1]*s, a[2]*s])

def quat_to_R(q):
    q = quat_norm(q)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

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
    return quat_norm(q)

def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def omega_mat(w):
    wx,wy,wz = w
    return np.array([[0,-wx,-wy,-wz],
                     [wx,0,wz,-wy],
                     [wy,-wz,0,wx],
                     [wz,wy,-wx,0]])

def integrate_attitude(q, w, dt):
    dq = (0.5*omega_mat(w) @ q) * dt
    qn = q + dq
    return quat_norm(qn)

def rand_unit(n):
    v = np.random.randn(n,3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)+1e-12
    return v

def ang_between_R(Ra, Rb):
    Rt = Ra @ Rb.T
    a = np.arccos(np.clip((np.trace(Rt)-1)/2, -1, 1))
    return a
