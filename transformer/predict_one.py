import torch
from model import TransformerPredictor
from utils import load_config
import os

def predict_once(model, input_sequence, device):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)  # [1, k, 4]
        predicted_bbox = model(input_tensor)  # [1, 4]
        return predicted_bbox.squeeze(0).cpu().numpy()

def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load model
    model = TransformerPredictor(
        input_dim=4,
        model_dim=config["hidden_dim"],  # reuse your GRU hidden_dim
        output_dim=4
    ).to(device)


    checkpoint_path = config["resume_from_checkpoint"]
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Checkpoint not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Example: Sequence of 6 previous bounding boxes [x, y, w, h]
    input_sequence = [
        [823,617,158,236],
        [828,612,138,238],
        [823,597,137,242],
        [828,589,129,242],
        [825,558,127,259]
    ]

    predicted_bbox = predict_once(model, input_sequence, device)
    print(f"Predicted bounding box: {predicted_bbox}")  

if __name__ == "__main__":
    main()

# [ 3.0603364   2.637174   -0.09712762 -0.07425666]