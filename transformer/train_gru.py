import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerPredictor
from dataset import DanceTrackDataset
from loss import CombinedLoss
from utils import load_config, save_checkpoint, load_checkpoint
import os
import heapq
import glob
import tqdm
from torch.optim.lr_scheduler import LambdaLR







config = load_config("config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = DanceTrackDataset("/DATA/Sonia/Datasets/DanceTrack/train", config["sequence_length"])
val_dataset = DanceTrackDataset("/DATA/Sonia/Datasets/DanceTrack/val", config["sequence_length"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

model = TransformerPredictor(
    input_dim=4,
    model_dim=config["hidden_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    output_dim=4
).to(device)

def get_linear_warmup_scheduler(optimizer, warmup_steps, max_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=1000, max_lr=1e-3)

TOP_K = 10
top_checkpoints = []
criterion = CombinedLoss(lambda_l1=5.0, lambda_ciou=2.0)

start_epoch = 0
if config.get("resume_from_checkpoint"):
    model, optimizer, start_epoch = load_checkpoint(
        model, optimizer, config["resume_from_checkpoint"], device
    )

for epoch in range(start_epoch, config["epochs"]):
    model.train()
    total_train_loss = 0
    total_samples = 0
    val_samples = 0
    train_loop = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config['epochs']}", unit="batch")
    for inputs, targets in train_loop:
        inputs, targets = inputs.to(device), targets.to(device)
        # print(f"inptuts shape: {inputs.shape}, targets shape: {targets.shape}")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item() * inputs.size(0)  # inputs.shape = [B, T, 4]
        total_samples += inputs.size(0)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        val_loop = tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config['epochs']}", unit="batch")
        for inputs, targets in val_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item() * inputs.size(0)
            val_samples += inputs.size(0)

    total_train_loss = total_train_loss / total_samples
    total_val_loss = total_val_loss / val_samples

    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss} | Val Loss: {total_val_loss}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_train_loss,
        'val_loss': total_val_loss
    }
    ckpt_path = os.path.join(config["save_dir"], f"epoch_{epoch+1}_train_loss_{total_train_loss}_val_loss_{total_val_loss}.pth")
    save_checkpoint(checkpoint, ckpt_path)

    heapq.heappush(top_checkpoints, (-total_val_loss, ckpt_path)) 

    if len(top_checkpoints) > TOP_K:
        worst = heapq.heappop(top_checkpoints)
        if os.path.exists(worst[1]):
            os.remove(worst[1])
            print(f"Deleted checkpoint: {worst[1]} (val_loss: {-worst[0]:.4f})")

