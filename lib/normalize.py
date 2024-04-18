import numpy as np


def normalize_pointcloud(pc: np.array) -> np.array:
    num_pts = pc.shape[0]
    centeroid = np.mean(pc, axis=0)
    pc = pc - centeroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m

    return pc