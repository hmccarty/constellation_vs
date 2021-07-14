from sim import Sim
from itertools import permutations, combinations
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Global defs
sim = Sim(headless=True)
constellation_size = 5


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def min_subset(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_idxs = []
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_idxs.append(np.argmax(distances))
        farthest_pts[i] = pts[farthest_idxs[-1]]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_idxs


def feature_to_pnt(feature, depth):
    pnt = feature - np.array([450. / 2., 450. / 2.])
    pnt[0] /= (543.19 * 0.001)
    pnt[1] /= (543.19 * 0.001)
    z = depth[int(feature[1]), int(feature[0])]
    pnt *= z
    return np.append(pnt, z)


def pnt_to_feature(pnt):
    feature = np.array([pnt[0], pnt[1]])
    if pnt[2] != 0:
        feature /= pnt[2]
    feature[0] *= (543.19 * 0.001)
    feature[1] *= (543.19 * 0.001)
    feature += np.array([450. / 2., 450. / 2.])
    return feature


def calculate_basis(pnts):
    a = pnts[1] - pnts[0]
    b = pnts[2] - pnts[0]

    # Reject if lines are colinear
    if np.linalg.norm(np.cross(a, b)) == 0.:
        return None

    # Use gram-schmidt process to second basis vector
    b -= (np.dot(b, a) / (np.dot(a, a))) * a

    # Take the cross product of current basis to find last basis vector
    c = np.cross(a, b)

    # Normalize to create orthonormal basis
    # a /= np.linalg.norm(a)
    # b /= np.linalg.norm(b)
    # c /= np.linalg.norm(c)

    return np.array([a, b, c]).T


def hash_3D(pnt):
    pnt[0] /= 0.005  # X bin size
    pnt[1] /= 0.005  # Y bin size
    pnt[2] /= 0.005  # Z bin size
    if not pnt.flags['C_CONTIGUOUS']:
        pnt = np.ascontiguousarray(pnt)
    return hash(pnt.astype(int).tostring())


def rigid_tf(a, b):
    """ Adapted from http://nghiaho.com/?page_id=671 """
    a = np.vstack(a).T
    b = np.vstack(b).T
    assert a.shape == b.shape

    # find mean column wise
    centroid_a = np.mean(a, axis=1)
    centroid_b = np.mean(b, axis=1)

    # ensure centroids are 3x1
    centroid_a = centroid_a.reshape(-1, 1)
    centroid_b = centroid_b.reshape(-1, 1)

    # subtract mean
    am = a - centroid_a
    bm = b - centroid_b

    H = am @ np.transpose(bm)

    # find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_a + centroid_b

    return R, t


# Receive frames
rgb, depth = sim.step()
cv.imshow("RGB: Init", rgb)
cv.waitKey()

# Extract features
kps, _ = cv.ORB_create().detectAndCompute(rgb, None)
kps = kps[-20:]
rgb_all_feats = cv.drawKeypoints(rgb, kps, None, (255, 0, 0), 4)
cv.imshow("RGB: Init", rgb_all_feats)
cv.waitKey()

# Select features
features = []
pnts = []
for i in range(len(kps)):
    feature = np.array(kps[i].pt)
    features.append(feature)
    pnts.append(feature_to_pnt(feature, depth))

idxs = min_subset(pnts, constellation_size + 1)
selected_kps = []
selected_features = []
selected_pnts = []
for i in idxs:
    selected_kps.append(kps[i])
    selected_features.append(features[i])
    selected_pnts.append(pnts[i])
rgb_selected_feats = cv.drawKeypoints(rgb, selected_kps, None, (255, 0, 0), 4)
cv.imshow("RGB: Init", rgb_selected_feats)
cv.waitKey()

# Hash and store
table = {}
basis_id = 0
stored_basis = {}
stored_framed_pnts = {}
for basis_pnts in permutations(selected_pnts, 3):
    # Calculate frame
    origin = basis_pnts[0]
    basis = calculate_basis(basis_pnts)

    if basis is not None:
        # Frame points in current basis
        framed_pnts = np.array(selected_pnts).T
        framed_pnts = (framed_pnts.T - origin).T
        framed_pnts = np.linalg.solve(basis, framed_pnts).T

        # Store each framed point
        pnt_id = 0
        for framed_pnt in framed_pnts:
            key = hash_3D(framed_pnt)
            if key in table:
                table[key].append({"basis_id": basis_id, "pnt_id": pnt_id})
            else:
                table[key] = [{"basis_id": basis_id, "pnt_id": pnt_id}]
            pnt_id += 1
        stored_basis[basis_id] = basis_pnts
        stored_framed_pnts[basis_id] = framed_pnts
        basis_id += 1
    else:
        # print("Frame could not be computed.")
        pass

print("Number of dictionary entries: ", len(table.keys()))

# Receive 10 new frames
for n in range(2):
    sim.applyVelocity(np.array([10., 0., 0., 0., 0., 0.]))
    update_rgb, update_depth = sim.step()
    cv.imshow("RGB: Update", update_rgb)
    cv.waitKey()

    # Extract features and convert
    kps, _ = cv.ORB_create().detectAndCompute(update_rgb, None)
    kps = kps[-20:]
    rgb_all_feats = cv.drawKeypoints(update_rgb, kps, None, (255, 0, 0), 4)
    cv.imshow("RGB: Update", rgb_all_feats)
    cv.waitKey()

    features = []
    pnts = []
    for i in range(len(kps)):
        feature = np.array(kps[i].pt)
        features.append(feature)
        pnts.append(feature_to_pnt(feature, depth))

    # Hash and vote
    vote = {}
    max_basis = None
    max_basis_pnts = None
    max_framed_pnts = None
    for basis_pnts in permutations(pnts, 3):
        origin = basis_pnts[0]
        basis = calculate_basis(basis_pnts)

        if basis is not None:
            # Frame points in current basis
            framed_pnts = np.array(pnts).T
            framed_pnts = (framed_pnts.T - origin).T
            framed_pnts = np.linalg.solve(basis, framed_pnts).T

            # Vote for each framed point
            for i in range(len(framed_pnts)):
                framed_pnt = framed_pnts[i]
                key = hash_3D(framed_pnt)
                if key in table:
                    entries = table[key]
                    for value in entries:
                        basis_id = value["basis_id"]
                        pnt_id = value["pnt_id"]
                        if basis_id in vote:
                            if pnt_id in vote[basis_id]:
                                # print("Point already registered.")
                                continue
                            else:
                                vote[basis_id][pnt_id] = pnts[i]
                        else:
                            vote[basis_id] = {pnt_id: pnts[i]}
                        if max_basis is None or \
                                len(vote[basis_id].values()) > len(vote[max_basis].values()):
                            max_basis = basis_id
                            max_basis_pnts = basis_pnts
                            max_framed_pnts = framed_pnts
            if len(vote[max_basis].values()) >= constellation_size:
                try:
                    pnts = []
                    for match in vote[max_basis].items():
                        pnts.append(match[1])
                    _, _ = rigid_tf(selected_pnts, pnts)
                except:
                    continue
                break
        else:
            # print("Frame could not be computed.")
            pass

    matched_features = [None] * constellation_size
    for match in vote[max_basis].items():
        matched_features[match[0]] = pnt_to_feature(match[1])

    # Draw basis on intial image
    orig_basis = stored_basis[max_basis]
    o_feature = pnt_to_feature(orig_basis[0])
    o_center = (int(o_feature[0]), int(o_feature[1]))
    a_feature = pnt_to_feature(orig_basis[1])
    a_center = (int(a_feature[0]), int(a_feature[1]))
    b_feature = pnt_to_feature(orig_basis[2])
    b_center = (int(b_feature[0]), int(b_feature[1]))
    rgb_orig_basis = rgb_selected_feats.copy()
    cv.line(rgb_orig_basis, o_center, a_center, (0, 255, 0), 4)
    cv.line(rgb_orig_basis, o_center, b_center, (125, 125, 0), 4)

    # Draw basis on final image
    o_feature = pnt_to_feature(max_basis_pnts[0])
    o_center = (int(o_feature[0]), int(o_feature[1]))
    a_feature = pnt_to_feature(max_basis_pnts[1])
    a_center = (int(a_feature[0]), int(a_feature[1]))
    b_feature = pnt_to_feature(max_basis_pnts[2])
    b_center = (int(b_feature[0]), int(b_feature[1]))
    cv.line(rgb_all_feats, o_center, a_center, (0, 255, 0), 4)
    cv.line(rgb_all_feats, o_center, b_center, (125, 125, 0), 4)

    for i in range(len(matched_features)):
        feature = matched_features[i]
        if feature is not None:
            center = (int(feature[0]), int(feature[1]))
            cv.circle(rgb_all_feats, center, 2, (0, 255, 0), 6)
            cv.putText(rgb_all_feats, str(i), center,
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (125, 125, 0), 2)

    for i in range(len(selected_features)):
        feature = selected_features[i]
        if feature is not None:
            center = (int(feature[0]), int(feature[1]))
            cv.circle(rgb_selected_feats, center, 2, (0, 255, 0), 6)
            cv.putText(rgb_selected_feats, str(i), center,
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (125, 125, 0), 2)

    cv.imshow("RGB: Update", rgb_all_feats)
    cv.imshow("RGB: Init", rgb_orig_basis)
    cv.waitKey()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(max_framed_pnts[:, 0], max_framed_pnts[:, 1], s=10,
                c='b', marker="s", label='first')
    selected_framed_pnts = stored_framed_pnts[max_basis]
    ax1.scatter(selected_framed_pnts[:, 0], selected_framed_pnts[:,
                1], s=10, c='r', marker="o", label='second')
    plt.legend(loc='upper left')
    plt.show()
    cv.waitKey()
    plt.close(fig)
cv.destroyAllWindows()
