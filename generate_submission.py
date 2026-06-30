import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        # Ensure correct sorting so we process images 0 to 1999 in order
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')], 
                                key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Extract integer ID from filename (e.g., '0.png' -> 0)
        image_id = int(os.path.splitext(img_name)[0])
        
        # Load 16-bit PNG image
        image = Image.open(img_path)
        image_np = np.array(image, dtype=np.float32) / 65535.0
        
        # Convert to tensor and shape to [C, H, W]
        image_tensor = torch.tensor(image_np)
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
            
        # Repeat channels if model expects RGB
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        return image_tensor, image_id

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load unlearned model
    checkpoint_path = 'data/depoisoned_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Did you run the training script?")
        return

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    cls_weight = state_dict.get('head.classification_head.cls_logits.weight')
    num_classes = 2
    if cls_weight is not None:
        num_classes = cls_weight.shape[0] // 9
        
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        num_classes=num_classes, 
        weights=None, 
        weights_backbone=None
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = TestDataset('data/test_set')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    results = []
    
    print("Starting inference on test set...")
    with torch.no_grad():
        for images, image_ids in data_loader:
            images = list(image.to(device) for image in images)
            
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                image_id = image_ids[i]
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Format: confidence x y width height
                pred_strings = []
                for box, score, label in zip(boxes, scores, labels):
                    # Filter by confidence > 0.2 as noted in evaluation metric
                    if score > 0.2:
                        x_min, y_min, x_max, y_max = box
                        x = x_min
                        y = y_min
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        pred_strings.append(f"{score:.4f} {x:.2f} {y:.2f} {width:.2f} {height:.2f}")
                
                pred_string = " ".join(pred_strings)
                results.append({'image_id': image_id, 'prediction_string': pred_string})
                
    # Create submission file
    df = pd.DataFrame(results)
    
    # Fill empty strings with a space character as required by Kaggle
    df['prediction_string'] = df['prediction_string'].apply(lambda x: " " if x == "" else x)
    
    # Sort by image_id just in case
    df = df.sort_values(by='image_id')
    df.to_csv('submission.csv', index=False)
    print("Saved submission file to submission.csv")

if __name__ == '__main__':
    main()
