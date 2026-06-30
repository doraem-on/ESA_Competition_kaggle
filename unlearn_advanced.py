import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import copy
import torch.nn as nn

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

        # Empty target for unlearning focal loss
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64)
        }
        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def prune_high_activations(model, data_loader, device, prune_ratio=0.01):
    """
    Surgically prunes the top highly activated neurons in the classification head 
    which are likely responsible for detecting the poison trigger.
    """
    print("Running forward pass to identify poison-trigger neurons for pruning...")
    model.eval()
    
    activations = []
    
    def hook_fn(m, i, o):
        # Average activation over spatial dimensions and batch
        activations.append(o.abs().mean(dim=(0, 2, 3)).cpu().detach())
        
    handles = []
    target_layer = None
    for name, module in model.head.classification_head.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
            handles.append(module.register_forward_hook(hook_fn))
            break # Just hook the first Conv2d in the cls head

    if target_layer is None:
        print("Warning: Could not find Conv2d in classification head to prune.")
        return model

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            model(images)
            
    for h in handles:
        h.remove()
        
    if not activations:
        return model
        
    mean_activations = torch.stack(activations).mean(dim=0)
    
    num_channels = mean_activations.shape[0]
    num_prune = max(1, int(num_channels * prune_ratio))
    _, top_indices = torch.topk(mean_activations, num_prune)
    
    print(f"Pruning top {num_prune}/{num_channels} channels in the classification head...")
    
    with torch.no_grad():
        target_layer.weight[top_indices, :, :, :] = 0.0
        if target_layer.bias is not None:
            target_layer.bias[top_indices] = 0.0
            
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = 'data/poisoned_model/poisoned_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

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

    # 1. High-Activation Pruning
    model = prune_high_activations(model, data_loader, device, prune_ratio=0.02) # 2% pruning

    # 2. Weight Anchoring (Simplified EWC) setup
    original_model = copy.deepcopy(model)
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 20
    lambda_l2 = 5000.0 # Aggressive weight anchor for clean performance preservation

    print(f"Starting advanced unlearning for {num_epochs} epochs with Weight Anchoring (lambda={lambda_l2})...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        unlearn_loss_total = 0
        anchor_loss_total = 0
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Unlearn Focal Loss
            loss_dict = model(images, targets)
            unlearn_loss = sum(loss for loss in loss_dict.values())

            # Calculate L2 distance to original weights
            l2_loss = 0.0
            for (name, param), (name_orig, param_orig) in zip(model.named_parameters(), original_model.named_parameters()):
                if param.requires_grad:
                    l2_loss += torch.sum((param - param_orig) ** 2)

            total_loss = unlearn_loss + (lambda_l2 * l2_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            unlearn_loss_total += unlearn_loss.item()
            anchor_loss_total += l2_loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Total: {epoch_loss/len(data_loader):.4f} | Unlearn: {unlearn_loss_total/len(data_loader):.4f} | Anchor L2: {anchor_loss_total/len(data_loader):.6f}")

    torch.save(model.state_dict(), 'data/advanced_depoisoned_model.pth')
    print("Saved advanced de-poisoned model to data/advanced_depoisoned_model.pth")

if __name__ == '__main__':
    main()
