import numpy as np


def rigid_tf(a, b):
    """ Adapted from http://nghiaho.com/?page_id=671 """
    a = np.vstack(a).T
    b = np.vstack(b).T
    assert a.shape == b.shape

    # Find mean column wise
    centroid_a = np.mean(a, axis=1)
    centroid_b = np.mean(b, axis=1)

    # Ensure centroids are 3x1
    centroid_a = centroid_a.reshape(-1, 1)
    centroid_b = centroid_b.reshape(-1, 1)

    # Subtract mean
    am = a - centroid_a
    bm = b - centroid_b

    H = am @ np.transpose(bm)

    # Find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_a + centroid_b

    return R, t
