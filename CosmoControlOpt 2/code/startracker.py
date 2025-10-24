
import numpy as np
from .wahba import wahba_attitude
from .utils import rand_unit
from .mekf import R_to_quat

class StarTracker:
    def __init__(self, n_stars_catalog=2000, fov_deg=20, dir_noise_std=1e-3):
        self.catalog = rand_unit(n_stars_catalog)
        self.fov = np.deg2rad(fov_deg)
        self.noise = dir_noise_std

    def observe(self, R_inertial_to_cam):
        cam_dirs = (R_inertial_to_cam @ self.catalog.T).T
        cosang = cam_dirs[:,2]/(np.linalg.norm(cam_dirs,axis=1)+1e-12)
        vis = np.logical_and(cam_dirs[:,2] > 0, np.arccos(np.clip(cosang, -1,1)) < self.fov)
        V = cam_dirs[vis]
        if V.shape[0] < 8:
            return None
        Vn = V + self.noise*np.random.randn(*V.shape)
        Vn = Vn/np.linalg.norm(Vn, axis=1, keepdims=True)
        idx = np.argsort(-Vn[:,2])[:20]
        obs = Vn[idx]
        ref = self.catalog[vis][idx]
        R_meas = wahba_attitude(obs, ref)
        q_meas = R_to_quat(R_meas)
        return q_meas
