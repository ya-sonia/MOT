import torch
import os
import numpy as np
from torch.utils.data import Dataset
import configparser

class DanceTrackDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5):
        self.sequence_length = sequence_length
        self.samples = []
        self._load_all(root_dir)

    def _load_all(self, root_dir):
        for video in sorted(os.listdir(root_dir)):
            video_path = os.path.join(root_dir, video)
            gt_path = os.path.join(video_path, "gt", "gt.txt")
            info_path = os.path.join(video_path, "seqinfo.ini")
            if not os.path.exists(gt_path) or not os.path.exists(info_path): continue

            config = configparser.ConfigParser()
            config.read(info_path)
            im_width = int(config["Sequence"]["imWidth"])
            im_height = int(config["Sequence"]["imHeight"])

            data = np.loadtxt(gt_path, delimiter=",", dtype=int)
            track_dict = {}

            for row in data:
                frame_id, track_id, x, y, w, h = row[:6]
                if track_id not in track_dict:
                    track_dict[track_id] = []
                # Normalize bbox
                norm_box = [x/im_width, y/im_height, w/im_width, h/im_height]
                track_dict[track_id].append((frame_id, norm_box))

            for track_id, traj in track_dict.items():
                traj.sort()
                traj = [bbox for _, bbox in traj]
                for i in range(len(traj) - self.sequence_length):
                    input_seq = traj[i:i+self.sequence_length]
                    target = traj[i+self.sequence_length]
                    self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
