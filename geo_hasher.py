import numpy as np
import cv2 as cv


class GeoHasher(object):
    def __init__(self, bin_size, thresh, constellation_size):
        self.num_frames = 0
        self.map = {}
        self.frames = {}
        self.bin_size = bin_size
        self.thresh = thresh
        self.constellation_size = constellation_size

    def calculate_frame(self, pnts):
        a = pnts[1] - pnts[0]
        b = pnts[2] - pnts[0]

        # Reject if lines are colinear
        if np.linalg.norm(np.cross(a, b)) == 0:
            return None, None

        # Use gram-schmidt process to second basis vector
        b -= (np.dot(b, a) / (np.dot(a, a))) * a

        # Take the cross product of current basis to find last basis vector
        c = np.cross(a, b)

        # Normalize to create orthonormal basis
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        c /= np.linalg.norm(c)

        return pnts[0], np.array([a, b, c]).T
        # return pnts[0], np.array([a, b]).T

    def store(self, origin, frame, pnts):
        # Store frame under specific id
        frame_id = self.num_frames
        self.frames[frame_id] = frame
        self.num_frames += 1

        pnts = (pnts.T - origin).T
        framed_pnts = np.linalg.solve(frame, pnts).T

        # Store framed pnt in 3d dictionary
        for i in range(len(framed_pnts)):
            pnt = framed_pnts[i]
            idx = self.index(pnt)
            if idx[0] in self.map:
                if idx[1] in self.map[idx[0]]:
                    if idx[2] in self.map[idx[0]][idx[1]]:
                        self.map[idx[0]][idx[1]][idx[2]].append([frame_id, i])
                    else:
                        self.map[idx[0]][idx[1]][idx[2]] = [[frame_id, i]]
                else:
                    self.map[idx[0]][idx[1]] = {idx[2]: [[frame_id, i]]}
            else:
                self.map[idx[0]] = {idx[1]: {idx[2]: [[frame_id, i]]}}

        return framed_pnts

    def index(self, pnt):
        return (pnt / self.bin_size).astype(int)

    def get(self, idx):
        if idx[0] in self.map and idx[1] in self.map[idx[0]] \
                and idx[2] in self.map[idx[0]][idx[1]]:
            return self.map[idx[0]][idx[1]][idx[2]]
        else:
            return None

    def vote(self, origin, frame, pnts):
        prev = {}
        result = {}
        max_frame = None

        pnts = (pnts.T - origin).T
        try:
            world = np.linalg.inv(frame)
            framed_pnts = np.linalg.solve(frame, pnts).T
        except np.linalg.LinAlgError:
            return None

        for pnt in framed_pnts:
            idx = self.index(pnt)
            if idx[0] in prev:
                if idx[1] in prev[idx[0]]:
                    if idx[2] in prev[idx[0]][idx[1]]:
                        continue
                    else:
                        prev[idx[0]][idx[1]][idx[2]] = True
                else:
                    prev[idx[0]][idx[1]] = {idx[2]: True}
            else:
                prev[idx[0]] = {idx[1]: {idx[2]: True}}

            # Fix this godfosaken mess that some may call code
            frames = self.get(idx)
            if frames is not None:
                pnt = np.linalg.solve(world, pnt)
                pnt += origin
                for frame in frames:
                    if frame[0] in result:
                        if result[frame[0]][0][frame[1]] is None:
                            result[frame[0]][1] += 1
                        result[frame[0]][0][frame[1]] = pnt
                    else:
                        result[frame[0]] = [
                            [None] * self.constellation_size, 1]
                        result[frame[0]][0][frame[1]] = pnt

                    if max_frame is None or \
                            result[frame[0]][1] > result[max_frame][1]:
                        max_frame = frame[0]

        if max_frame is not None:
            return result[max_frame][0], result[max_frame][1]
        return None

    def clear(self):
        self.num_frames = 0
        self.map = {}
        self.frames = {}

    def is_empty(self):
        if self.num_frames == 0:
            return True
        else:
            return False
