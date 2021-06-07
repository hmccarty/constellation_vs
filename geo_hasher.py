import numpy as np


class GeoHasher(object):
    def __init__(self, bin_size, thresh):
        self.num_frames = 0
        self.map = {}
        self.frames = {}
        self.bin_size = bin_size
        self.thresh = thresh

    def calculate_frame(self, pnts):
        a = pnts[1] - pnts[0]
        b = pnts[2] - pnts[0]

        # Reject if lines are colinear
        if np.linalg.norm(np.cross(a, b)) == 0:
            return None, None

        # Use gram-schmidt process to second basis vector
        b -= (np.dot(b, a) / (np.dot(a, a))) * a

        # Take the cross product of current basis to find last basis vector
        # c = np.cross(a, b)

        # Normalize to create orthonormal basis
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        # c /= np.linalg.norm(c)

        # return np.array([a, b, c]).T
        return pnts[0], np.array([a, b]).T

    def store(self, origin, frame, pnts):
        # Store frame under specific id
        frame_id = self.num_frames
        self.frames[frame_id] = frame
        self.num_frames += 1
        print(pnts.shape)
        print(origin.shape)

        pnts = (pnts.T - origin).T
        framed_pnts = np.linalg.solve(frame, pnts).T

        # Store framed pnt in 3d dictionary
        # for pnt in framed_pnts:
        #     idx = (pnt / self.bin_size).astype(int)
        #     if idx[0] in self.map:
        #         if idx[1] in self.map[idx[0]]:
        #             if idx[2] in self.map[idx[0]][idx[1]]:
        #                 self.map[idx[0]][idx[1]][idx[2]].append(frame_id)
        #             else:
        #                 self.map[idx[0]][idx[1]][idx[2]] = [frame_id]
        #         else:
        #             self.map[idx[0]][idx[1]] = {idx[2]: [frame_id]}
        #     else:
        #         self.map[idx[0]] = {idx[1]: {idx[2]: [frame_id]}}

        for pnt in framed_pnts:
            idx = (pnt / self.bin_size).astype(int)
            if idx[0] in self.map:
                if idx[1] in self.map[idx[0]]:
                    self.map[idx[0]][idx[1]].append(frame_id)
                else:
                    self.map[idx[0]][idx[1]] = [frame_id]
            else:
                self.map[idx[0]] = {idx[1]: [frame_id]}

        return framed_pnts

    def get(self, pnt):
        idx = (pnt / self.bin_size).astype(int)
        # if idx[0] in self.map and idx[1] in self.map[idx[0]] \
        #         and idx[2] in self.map[idx[0]][idx[1]]:
        #     return self.map[idx[0]][idx[1]][idx[2]]
        # else:
        #     return None

        if idx[0] in self.map and idx[1] in self.map[idx[0]]:
            return self.map[idx[0]][idx[1]]
        else:
            return None

    def vote(self, origin, frame, pnts):
        result = {}
        max_frame = None

        pnts = (pnts.T - origin).T
        try:
            world = np.linalg.inv(frame)
            framed_pnts = np.linalg.solve(frame, pnts).T
        except np.linalg.LinAlgError:
            return None, None

        print(framed_pnts.shape)
        for pnt in framed_pnts:
            frames = self.get(pnt)
            if frames is not None:
                pnt = np.linalg.solve(world, pnt)
                pnt += origin
                for frame in frames:
                    if frame in result:
                        result[frame].append(pnt)
                    else:
                        result[frame] = [pnt]

                    if max_frame is None or \
                            len(result[frame]) > len(result[max_frame]):
                        max_frame = frame

        if max_frame is not None:
            return result[max_frame], framed_pnts
        return None, framed_pnts

    def clear(self):
        self.num_frames = 0
        self.map = {}
        self.frames = {}

    def is_empty(self):
        if self.num_frames == 0:
            return True
        else:
            return False
