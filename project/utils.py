import yaml
import os
import torch

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
    return model, optimizer, start_epoch


def calculate_iou(box1, box2):
    # box: Tensor of shape [4] with [cx, cy, w, h]
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    x1_min = cx1 - w1 / 2
    y1_min = cy1 - h1 / 2
    x1_max = cx1 + w1 / 2
    y1_max = cy1 + h1 / 2

    x2_min = cx2 - w2 / 2
    y2_min = cy2 - h2 / 2
    x2_max = cx2 + w2 / 2
    y2_max = cy2 + h2 / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return (inter_area / union_area) if union_area > 0 else 0.0

def calculate_center_distance(box1, box2):
    # box: Tensor of shape [4] with [cx, cy, w, h]
    cx1, cy1 = box1[0], box1[1]
    cx2, cy2 = box2[0], box2[1]
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2).sqrt().item()
