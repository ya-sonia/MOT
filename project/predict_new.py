# /DATA/Sonia/MOT_Datasets/DanceTrack/val/dancetrack0004

import torch
import os
from model import GRUPredictor
from utils import load_config
import configparser  # To read sequence.ini

def normalize_bbox(bbox, im_width, im_height):
    x, y, w, h = bbox
    return [x / im_width, y / im_height, w / im_width, h / im_height]

def denormalize_bbox(bbox, im_width, im_height):
    x, y, w, h = bbox
    return [x * im_width, y * im_height, w * im_width, h * im_height]

def read_image_size(seqinfo_ini_path):
    config = configparser.ConfigParser()
    config.read(seqinfo_ini_path)
    im_width = int(config["Sequence"]["imWidth"])
    im_height = int(config["Sequence"]["imHeight"])
    return im_width, im_height

def predict_once(model, input_sequence, im_width, im_height, device):
    model.eval()
    with torch.no_grad():
        # Normalize input sequence
        norm_input = [normalize_bbox(b, im_width, im_height) for b in input_sequence]
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(device)  # shape [1, k, 4]
        output = model(input_tensor)  # shape [1, 4]
        predicted_bbox = output.squeeze(0).cpu().numpy().tolist()
        return denormalize_bbox(predicted_bbox, im_width, im_height)

def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GRUPredictor(
        input_dim=4,
        hidden_dim=config["hidden_dim"],
        output_dim=4
    ).to(device)

    checkpoint_path = config["resume_from_checkpoint"]
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Checkpoint not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    input_sequence = [
        [823, 617, 158, 236],
        [828, 612, 138, 238],
        [823, 597, 137, 242],
        [828, 589, 129, 242],
        [825, 558, 127, 259]
    ]

    sequence_ini_path = "/DATA/Sonia/Datasets/DanceTrack/val/dancetrack0004/seqinfo.ini"
    im_width, im_height = read_image_size(sequence_ini_path)

    predicted_bbox = predict_once(model, input_sequence, im_width, im_height, device)

    print(f"Predicted Bounding Box (pixel space): {predicted_bbox}")

if __name__ == "__main__":
    main()

