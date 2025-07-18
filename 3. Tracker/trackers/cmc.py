import pickle
import numpy as np

class CMC:
    def __init__(self, vid_name):
        super(CMC, self).__init__()

        if 'MOT17' in vid_name:
            vid_name = vid_name.split('-FRCNN')[0]
        elif 'dance' in vid_name.lower():
            vid_name = 'dancetrack-' + vid_name.split('dancetrack')[1]

        self.warp_dict = {}
        self._load_gmc_file('./trackers/cmc/GMC-' + vid_name + '.txt')

    def _load_gmc_file(self, path):
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split('\t')
                if len(tokens) < 7:
                    continue
                frame_id = int(tokens[0])
                warp_matrix = np.array([
                    [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                    [float(tokens[4]), float(tokens[5]), float(tokens[6])]
                ], dtype=np.float32)
                self.warp_dict[frame_id] = warp_matrix

    def get_warp_matrix(self, frame_id):
        return self.warp_dict.get(frame_id, np.eye(2, 3, dtype=np.float32))



def apply_cmc(tracks, frame_id, cmc_obj):
    """
    Apply global motion compensation to all GRU-based tracks.
    """
    if len(tracks) == 0:
        return

    warp_matrix = cmc_obj.get_warp_matrix(frame_id)
    rot = warp_matrix[:, :2]
    trans = warp_matrix[:, 2]

    for t in tracks:
        if t.mean is None:
            continue
        cx, cy, w, h = t.mean
        cx_new, cy_new = rot @ np.array([cx, cy]) + trans
        t.mean = np.array([cx_new, cy_new, w, h])
