import numpy as np


class GeoHasher(object):
    def __init__(self, bin_size, thresh):
        self.num_frames = 0
        self.map = {}
        self.frames = {}
        self.bin_size = bin_size
        self.thresh = thresh

    def store(self, frame, pnts):
        # Store frame under specific id
        frame_id = self.num_frames
        self.frames[frame_id] = frame
        self.num_frames += 1

        framed_pnts = np.linalg.solve(frame, pnts).T

        # Store framed pnt in 3d dictionary
        for pnt in framed_pnts:
            idx = (pnt / self.bin_size).astype(int)
            if idx[0] in self.map:
                if idx[1] in self.map[idx[0]]:
                    if idx[2] in self.map[idx[0]][idx[1]]:
                        self.map[idx[0]][idx[1]][idx[2]].append(frame_id)
                    else:
                        self.map[idx[0]][idx[1]][idx[2]] = [frame_id]
                else:
                    self.map[idx[0]][idx[1]] = {idx[2]: [frame_id]}
            else:
                self.map[idx[0]] = {idx[1]: {idx[2]: [frame_id]}}

    def get(self, pnt):
        idx = (pnt / self.bin_size).astype(int)
        if idx[0] in self.map and idx[1] in self.map[idx[0]] \
                and idx[2] in self.map[idx[0]][idx[1]]:
            return self.map[idx[0]][idx[1]][idx[2]]
        else:
            return None

    def vote(self, frame, pnts):
        result = {}
        max_frame = None

        try:
            world = np.linalg.inv(frame)
            framed_pnts = np.linalg.solve(frame, pnts).T
        except np.linalg.LinAlgError:
            return None, None

        frame_list = []
        for pnt in frame.T:
            frame_list.append(pnt)

        for pnt in framed_pnts:
            frames = self.get(pnt)
            if frames is not None:
                pnt = np.linalg.solve(world, pnt)
                for frame in frames:
                    if frame in result:
                        result[frame].append(pnt)
                    else:
                        result[frame] = [pnt]

                    if len(result[frame]) > self.thresh:
                        return frame_list, result[frame]
                    elif max_frame is None or \
                            len(result[frame]) > len(result[max_frame]):
                        max_frame = frame

        if max_frame is not None:
            return frame_list, result[max_frame]
        return None, None

    def clear(self):
        self.num_frames = 0
        self.map = {}
        self.frames = {}

    def is_empty(self):
        if self.num_frames == 0:
            return True
        else:
            return False
