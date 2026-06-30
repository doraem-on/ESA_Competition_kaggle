import os
import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd
from torchvision.ops import nms

class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image_id = int(os.path.splitext(img_name)[0])
        
        image = Image.open(img_path)
        image_np = np.array(image, dtype=np.float32) / 65535.0
        
        image_tensor = torch.tensor(image_np)
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        return image_tensor, image_id

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['data/advanced_depoisoned_model.pth'], help='Paths to model checkpoints for ensembling')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--conf-thresh', type=float, default=0.2, help='Confidence threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models = []
    for path in args.models:
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
            
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        cls_weight = state_dict.get('head.classification_head.cls_logits.weight')
        num_classes = cls_weight.shape[0] // 9 if cls_weight is not None else 2
        
        model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=num_classes, weights=None, weights_backbone=None)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
        
    print(f"Loaded {len(models)} models for ensembling.")

    dataset = TestDataset('data/test_set')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    results = []
    
    print("Starting ensemble inference...")
    with torch.no_grad():
        for images, image_ids in data_loader:
            images = list(img.to(device) for img in images)
            
            batch_boxes = [[] for _ in range(len(images))]
            batch_scores = [[] for _ in range(len(images))]
            batch_labels = [[] for _ in range(len(images))]
            
            for model in models:
                outputs = model(images)
                for i, output in enumerate(outputs):
                    batch_boxes[i].append(output['boxes'])
                    batch_scores[i].append(output['scores'])
                    batch_labels[i].append(output['labels'])
                    
            for i, image_id in enumerate(image_ids):
                if not batch_boxes[i]:
                    results.append({'image_id': image_id, 'prediction_string': " "})
                    continue
                    
                boxes = torch.cat(batch_boxes[i])
                scores = torch.cat(batch_scores[i])
                labels = torch.cat(batch_labels[i])
                
                if boxes.shape[0] > 0:
                    keep = nms(boxes, scores, args.iou_thresh)
                    boxes = boxes[keep].cpu().numpy()
                    scores = scores[keep].cpu().numpy()
                    labels = labels[keep].cpu().numpy()
                else:
                    boxes, scores, labels = [], [], []
                
                pred_strings = []
                for box, score, label in zip(boxes, scores, labels):
                    if score > args.conf_thresh:
                        x_min, y_min, x_max, y_max = box
                        pred_strings.append(f"{score:.4f} {x_min:.2f} {y_min:.2f} {x_max - x_min:.2f} {y_max - y_min:.2f}")
                
                pred_string = " ".join(pred_strings)
                results.append({'image_id': image_id, 'prediction_string': pred_string if pred_string else " "})
                
    df = pd.DataFrame(results).sort_values(by='image_id')
    df.to_csv('submission.csv', index=False)
    print("Saved ensembled submission file to submission.csv")

if __name__ == '__main__':
    main()
