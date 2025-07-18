import torch
from collections import deque
from trackers.model import GRUPredictor

class GRUMotionModel:
    def __init__(self, sequence_length=5, device='cpu', im_width=1920, im_height=1080):
        """
        model: trained PyTorch GRU model
        sequence_length: how many past steps to use for prediction
        device: 'cuda' or 'cpu'
        im_width, im_height: used for (de)normalizing bboxes
        """
        self.model = GRUPredictor(input_dim=4, hidden_dim=64, output_dim=4)
        checkpoint = torch.load('/DATA/Sonia/TrackTrack/3. Tracker/trackers/gru_model.pth', map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.sequence_length = sequence_length
        self.device = device
        self.im_width = im_width
        self.im_height = im_height

        self.track_histories = {}

    def normalize_bbox(self, bbox):
        x, y, w, h = bbox
        return [x / self.im_width, y / self.im_height, w / self.im_width, h / self.im_height]

    def denormalize_bbox(self, bbox):
        x, y, w, h = bbox
        return [x * self.im_width, y * self.im_height, w * self.im_width, h * self.im_height]

    def initiate(self, track_id, initial_bbox):
        """
        Initialize the object history with initial bbox: [x, y, w, h]
        """
        history = deque(maxlen=self.sequence_length)
        for _ in range(self.sequence_length - 1):
            history.append(initial_bbox)
        history.append(initial_bbox)
        self.track_histories[track_id] = history
        return initial_bbox, None  

    def predict(self, track_id):
        """
        Use GRU to predict the next bbox for the given track.
        Returns predicted [x, y, w, h] in pixel space.
        """
        if track_id not in self.track_histories:
            raise ValueError(f"Track ID {track_id} not initialized.")
        history = self.track_histories[track_id]
        if len(history) < self.sequence_length:
            return history[-1]  # Not enough context

        # Normalize history
        norm_seq = [self.normalize_bbox(bbox) for bbox in history]
        input_seq = torch.tensor([norm_seq], dtype=torch.float32).to(self.device)  # [1, seq_len, 4]

        with torch.no_grad():
            pred = self.model(input_seq).cpu().numpy()[0]  # [4]
            pred_bbox = self.denormalize_bbox(pred)
        return pred_bbox

    def update(self, track_id, new_measurement):
        """
        Update the history with new detection bbox [x, y, w, h] (in pixel space)
        """
        if track_id not in self.track_histories:
            self.initiate(track_id, new_measurement)
        else:
            self.track_histories[track_id].append(new_measurement)
        return self.track_histories[track_id][-1], None

    def project(self, track_id):
        """
        Return current predicted state (last predicted bbox in pixel space).
        If not enough history, return last bbox.
        """
        if track_id not in self.track_histories:
            raise ValueError(f"Track ID {track_id} not initialized.")
        return self.predict(track_id), None
