import torch
import torch.nn as nn

def bbox_ciou(pred, target):
    # Convert top-left format [x, y, w, h] → corners
    pred_x1 = pred[:, 0]
    pred_y1 = pred[:, 1]
    pred_x2 = pred[:, 0] + pred[:, 2]
    pred_y2 = pred[:, 1] + pred[:, 3]

    target_x1 = target[:, 0]
    target_y1 = target[:, 1]
    target_x2 = target[:, 0] + target[:, 2]
    target_y2 = target[:, 1] + target[:, 3]

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + 1e-7

    iou = inter_area / union_area

    # Centers
    pred_cx = pred[:, 0] + pred[:, 2] / 2
    pred_cy = pred[:, 1] + pred[:, 3] / 2
    target_cx = target[:, 0] + target[:, 2] / 2
    target_cy = target[:, 1] + target[:, 3] / 2

    center_dist = (pred_cx - target_cx)**2 + (pred_cy - target_cy)**2

    # Enclosing box
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)
    enc_diag = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + 1e-7

    # Aspect ratio penalty (v)
    pred_w, pred_h = pred[:, 2], pred[:, 3]
    target_w, target_h = target[:, 2], target[:, 3]
    v = (4 / (torch.pi**2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h))**2

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_dist / enc_diag) - alpha * v
    loss_ciou = 1 - ciou

    return loss_ciou.mean()


class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ciou=1.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ciou = lambda_ciou
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ciou = bbox_ciou(pred, target)
        return self.lambda_l1 * l1  + self.lambda_ciou * ciou
