import torch
from torch.utils.data import DataLoader
from model import TransformerPredictor
from dataset import DanceTrackDataset
from loss import CombinedLoss
from utils import load_config, calculate_iou, calculate_center_distance
import os
import tqdm

def evaluate(model, loader, device, iou_threshold=0.8, dist_threshold=5):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    iou_correct = 0
    dist_correct = 0
    criterion = CombinedLoss(lambda_l1=5.0, lambda_ciou=2.0)
    with torch.no_grad():
        test_loop = tqdm.tqdm(loader, desc="Evaluating", unit="batch")
        for inputs, targets in test_loop:
            inputs, targets = inputs.to(device), targets.to(device)  # [B, k, 4] and [B, 4]
            outputs = model(inputs)  # [B, 4]
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            for pred_box, gt_box in zip(outputs, targets):
                iou = calculate_iou(pred_box, gt_box)
                dist = calculate_center_distance(pred_box, gt_box)

                if iou >= iou_threshold:
                    iou_correct += 1
                if dist < dist_threshold:
                    dist_correct += 1

    avg_loss = total_loss / total_samples
    iou_acc = iou_correct / total_samples
    dist_acc = dist_correct / total_samples
    return avg_loss, iou_acc, dist_acc

def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load model
    model = TransformerPredictor(
        input_dim=4,
        model_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        output_dim=4
    ).to(device)


    checkpoint_path = config["resume_from_checkpoint"]
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Checkpoint not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Load validation data
    val_dataset = DanceTrackDataset(
        root_dir="/DATA/Sonia/Datasets/DanceTrack/val",
        sequence_length=config["sequence_length"]
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Evaluate
    val_loss, iou_acc, dist_acc = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"IoU@0.8 Accuracy: {iou_acc * 100:.2f}%")
    print(f"Center Distance < 10 Accuracy: {dist_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
