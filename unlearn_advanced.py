import os
import argparse
import copy
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class UnlearnDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)
        image_np = np.array(image, dtype=np.float32) / 65535.0
        
        image_tensor = torch.tensor(image_np)
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64)
        }
        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def prune_taylor_expansion(model, data_loader, device, prune_ratio):
    print("Running forward and backward pass to identify poison-trigger neurons via ActivationxGradient...")
    model.train() # Must be in train mode for backward pass
    
    activations = []
    
    def hook_fn(m, i, o):
        o.retain_grad()
        activations.append(o)
        
    target_layer = None
    for name, module in model.head.classification_head.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
            h = module.register_forward_hook(hook_fn)
            break

    if target_layer is None:
        print("Warning: Could not find Conv2d in classification head to prune.")
        return model

    model.zero_grad()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())
        loss.backward()
            
    h.remove()
    
    scores = torch.zeros(target_layer.out_channels, device=device)
    for act in activations:
        if act.grad is not None:
            # Taylor Expansion Score: |Activation * Gradient|
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

    checkpoint_path = 'data/poisoned_model/poisoned_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    cls_weight = state_dict.get('head.classification_head.cls_logits.weight')
    num_classes = cls_weight.shape[0] // 9 if cls_weight is not None else 2

    model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=num_classes, weights=None, weights_backbone=None)
    model.load_state_dict(state_dict)
    model.to(device)

    dataset = UnlearnDataset('data/unlearn_set')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 1. ActivationxGradient Pruning
    model = prune_taylor_expansion(model, data_loader, device, prune_ratio=args.prune_ratio)

    # 2. Architecture Freezing
    print("Freezing Backbone and Regression Head...")
    for param in model.backbone.body.parameters():
        param.requires_grad = False
    for param in model.head.regression_head.parameters():
        param.requires_grad = False
    # BN layers in backbone are frozen, FPN and Classification Head remain trainable

    # 3. Weight Anchoring Setup
    original_model = copy.deepcopy(model)
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss, unlearn_loss_total, anchor_loss_total = 0, 0, 0
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
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

    torch.save(model.state_dict(), args.output_model)
    print(f"Saved model to {args.output_model}")

if __name__ == '__main__':
    main()
