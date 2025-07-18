import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import GRUPredictor
from dataset import DanceTrackDataset
from utils import load_config, save_checkpoint, load_checkpoint
import os
import heapq
import glob
import tqdm



config = load_config("config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = DanceTrackDataset("/DATA/Sonia/Datasets/DanceTrack/train", config["sequence_length"])
val_dataset = DanceTrackDataset("/DATA/Sonia/Datasets/DanceTrack/val", config["sequence_length"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

model = GRUPredictor(input_dim=4, hidden_dim=config["hidden_dim"], output_dim=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
TOP_K = 10
top_checkpoints = []
criterion = nn.MSELoss()

start_epoch = 0
if config.get("resume_from_checkpoint"):
    model, optimizer, start_epoch = load_checkpoint(
        model, optimizer, config["resume_from_checkpoint"], device
    )

for epoch in range(start_epoch, config["epochs"]):
    model.train()
    total_train_loss = 0
    train_loop = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config['epochs']}", unit="batch")
    for inputs, targets in train_loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        val_loop = tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config['epochs']}", unit="batch")
        for inputs, targets in val_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

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

