import os
import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

try:
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2 import model_zoo
except ImportError:
    print("Warning: detectron2 is not installed. Script is adapted for Kaggle environments.")

from torch.utils.data import Dataset, DataLoader

class UnlearnDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)
        image_np = np.array(image, dtype=np.float32)
        
        # Convert to C, H, W
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=0)
        if image_np.shape[0] == 1:
            image_np = np.repeat(image_np, 3, axis=0)
            
        from detectron2.structures import Instances, Boxes
        image_tensor = torch.tensor(image_np)
        
        # Provide empty instances to force unlearning
        target = Instances((image_tensor.shape[1], image_tensor.shape[2]))
        target.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
        target.gt_classes = torch.zeros(0, dtype=torch.int64)
        
        return {"image": image_tensor, "instances": target}

def collate_fn(batch):
    return batch

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.RETINANET.NUM_CLASSES = 1 # Update if competition requires more classes
    return cfg

def prune_taylor_expansion(model, data_loader, device, prune_ratio):
    print("Running forward and backward pass to identify poison-trigger neurons via ActivationxGradient...")
    model.train()
    activations = []
    
    def hook_fn(m, i, o):
        o.retain_grad()
        activations.append(o)
        
    target_layer = None
    if hasattr(model, 'head') and hasattr(model.head, 'cls_subnet'):
        for name, module in model.head.cls_subnet.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                h = target_layer.register_forward_hook(hook_fn)
                break

    if target_layer is None:
        print("Warning: Could not find Conv2d in classification subnet to prune.")
        return model

    model.zero_grad()
    for batch in data_loader:
        for d in batch:
            d["image"] = d["image"].to(device)
            d["instances"] = d["instances"].to(device)
            
        loss_dict = model(batch)
        loss = sum(l for l in loss_dict.values())
        loss.backward()
            
    h.remove()
    
    scores = torch.zeros(target_layer.out_channels, device=device)
    for act in activations:
        if act.grad is not None:
            score = (act * act.grad).abs().mean(dim=(0, 2, 3))
            scores += score
            
    num_channels = scores.shape[0]
    num_prune = max(1, int(num_channels * prune_ratio))
    _, top_indices = torch.topk(scores, num_prune)
    
    print(f"Pruning top {num_prune}/{num_channels} channels based on ActivationxGradient...")
    
    with torch.no_grad():
        target_layer.weight[top_indices, :, :, :] = 0.0
        if target_layer.bias is not None:
            target_layer.bias[top_indices] = 0.0
            
    model.zero_grad()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune-ratio', type=float, default=0.01)
    parser.add_argument('--lambda-l2', type=float, default=1000.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output-model', type=str, default='data/advanced_depoisoned_model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Configuration: prune_ratio={args.prune_ratio}, lambda_l2={args.lambda_l2}, lr={args.lr}, epochs={args.epochs}")

    cfg = setup_cfg()
    model = build_model(cfg)
    
    checkpoint_path = 'data/poisoned_model/poisoned_model.pth'
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    model.to(device)

    dataset = UnlearnDataset('data/unlearn_set')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 1. ActivationxGradient Pruning
    model = prune_taylor_expansion(model, data_loader, device, prune_ratio=args.prune_ratio)

    # 2. Architecture Freezing
    print("Freezing Backbone and Regression Head...")
    for param in model.backbone.bottom_up.parameters():
        param.requires_grad = False
    for param in model.head.bbox_subnet.parameters():
        param.requires_grad = False
    if hasattr(model.head, 'bbox_pred'):
        for param in model.head.bbox_pred.parameters():
            param.requires_grad = False

    # 3. Weight Anchoring Setup
    original_model = copy.deepcopy(model)
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss, unlearn_loss_total, anchor_loss_total = 0, 0, 0
        
        for batch in data_loader:
            for d in batch:
                d["image"] = d["image"].to(device)
                d["instances"] = d["instances"].to(device)

            loss_dict = model(batch)
            unlearn_loss = sum(loss for loss in loss_dict.values())

            l2_loss = 0.0
            for (name, param), (name_orig, param_orig) in zip(model.named_parameters(), original_model.named_parameters()):
                if param.requires_grad:
                    l2_loss += torch.sum((param - param_orig) ** 2)

            total_loss = unlearn_loss + (args.lambda_l2 * l2_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            unlearn_loss_total += unlearn_loss.item()
            anchor_loss_total += l2_loss.item()
            
        print(f"Epoch [{epoch+1}/{args.epochs}], Total: {epoch_loss/len(data_loader):.4f} | Unlearn: {unlearn_loss_total/len(data_loader):.4f} | Anchor L2: {anchor_loss_total/len(data_loader):.6f}")

    # Save only the state dict for Detectron2 compatibility
    torch.save({"model": model.state_dict()}, args.output_model)
    print(f"Saved model to {args.output_model}")

if __name__ == '__main__':
    main()
