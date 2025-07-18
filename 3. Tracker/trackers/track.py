import numpy as np
from trackers.utils import get_prev_box
from trackers.gru import GRUMotionModel
from collections import deque
import torch

def get_vel(b_1, b_2):
    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[0]**2 + deltas[1]**2) + 1e-5
    norm_lb = np.sqrt(deltas[0]**2 + deltas[3]**2) + 1e-5
    norm_rt = np.sqrt(deltas[2]**2 + deltas[1]**2) + 1e-5
    norm_rb = np.sqrt(deltas[2]**2 + deltas[3]**2) + 1e-5

    # Get velocities
    vel_lt = np.array([b_2[0] - b_1[0], b_2[1] - b_1[1]]) / norm_lt
    vel_lb = np.array([b_2[0] - b_1[0], b_2[3] - b_1[1]]) / norm_lb
    vel_rt = np.array([b_2[2] - b_1[2], b_2[1] - b_1[1]]) / norm_rt
    vel_rb = np.array([b_2[2] - b_1[2], b_2[3] - b_1[1]]) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=0)

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class TrackCounter(object):
    track_count = 0

    def get_track_id(self):
        self.track_count += 1
        return self.track_count

class BaseTrack(object):
    track_id = 0
    end_frame_id = 0
    state = TrackState.New

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

class Track(BaseTrack):
    def __init__(self, args, detection, device='cuda'):
        self.args = args
        self.box = detection[:4]  # x1y1x2y2
        self.score = detection[4]
        self.device = device
        self.sequence_length = args.sequence_length
        self.delta_t = 3  
        self.gru_model = GRUMotionModel(sequence_length=self.sequence_length, device=device, im_width=args.img_w, im_height=args.img_h)

        self.track_id = None
        self.history = deque(maxlen=self.sequence_length)  # stores [cx, cy, w, h]
        self.mean = None  # [cx, cy, w, h]
        self.predicted_mean = None  # Store GRU prediction separately
        self.velocity = np.zeros((4, 2))

        # Appearance feature
        self.alpha = 0.95
        self.feat = detection[6:][np.newaxis, :].copy()

        self.end_frame_id = 0
        self.state = TrackState.New

    def update_features(self, feat, score):
        beta = self.alpha + (1 - self.alpha) * (1 - score)
        self.feat = beta * self.feat + (1 - beta) * feat
        self.feat /= np.linalg.norm(self.feat)

    def initiate(self, frame_id, counter):
        self.track_id = counter.get_track_id()
        initial_cxcywh = self.cxcywh.copy()
        print(f"Initiating Track ID {self.track_id} at frame {frame_id} with box {self.box} and score {self.score}")
        # Initialize both local history and GRU model's history
        for i in range(self.sequence_length - 1):
            shifted = initial_cxcywh.copy()
            shifted[0] -= i * 1.0
            shifted[1] -= i * 1.0
            self.history.appendleft(shifted)
        self.history.append(initial_cxcywh)
        
        # Initialize GRU model's history
        self.gru_model.initiate(self.track_id, initial_cxcywh)
        
        self.mean = initial_cxcywh.copy()
        self.end_frame_id = frame_id
        self.state = TrackState.New

    def predict(self):
        if len(self.history) < self.sequence_length:
            return

        # Use GRU model for prediction
        predicted_bbox = self.gru_model.predict(self.track_id)
        self.predicted_mean = predicted_bbox.copy()
        
        # Update local history with prediction (for next frame)
        self.history.append(predicted_bbox)

    def update(self, frame_id, detection):
        detection_cxcywh = self._to_cxcywh(detection.box)

        self.mean = detection_cxcywh.copy()  # Set mean only from actual detection
        self.predicted_mean = None           # Reset prediction on update

        self.history.append(detection_cxcywh)
        self.gru_model.update(self.track_id, detection_cxcywh)

        self.prev_score = self.score
        self.box = detection.box.copy()
        self.score = detection.score 

        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, d_t)
            if prev_box is not None:
                self.velocity += get_vel(prev_box, detection.box)

        self.end_frame_id = frame_id
        self.state = TrackState.Tracked

    @property
    def cxcywh(self):
        if self.predicted_mean is not None:
            return self.predicted_mean.copy()
        elif self.mean is not None:
            return self.mean.copy()
        else:
            x1, y1, x2, y2 = self.box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            return np.array([cx, cy, w, h])

    @property
    def x1y1x2y2(self):
        cx, cy, w, h = self.cxcywh  
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])

    def _to_cxcywh(self, box):
        box = np.array(box) 
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h])

    @property
    def x1y1wh(self):
        """Returns bounding box in [x1, y1, width, height] format.
        
        Returns:
            np.array: [x1, y1, width, height] bounding box coordinates
        """
        if self.mean is None:
            # Use raw box coordinates if mean is not available
            x1, y1 = self.box[0], self.box[1]
            w = self.box[2] - self.box[0]  # Calculate width
            h = self.box[3] - self.box[1]  # Calculate height
        else:
            # Use mean (center) coordinates if available
            x1 = self.mean[0] - self.mean[2] / 2  # Calculate x1 from center
            y1 = self.mean[1] - self.mean[3] / 2  # Calculate y1 from center
            w, h = self.mean[2], self.mean[3]     # Use stored width/height

        return np.array([x1, y1, w, h])