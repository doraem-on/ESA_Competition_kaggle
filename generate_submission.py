import os
import argparse
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torchvision.ops import nms

try:
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2 import model_zoo
except ImportError:
    print("Warning: detectron2 is not installed. Script is adapted for Kaggle environments.")

from torch.utils.data import Dataset, DataLoader

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
        image_np = np.array(image, dtype=np.float32)
        
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=0)
        if image_np.shape[0] == 1:
            image_np = np.repeat(image_np, 3, axis=0)

        image_tensor = torch.tensor(image_np)
        return {"image": image_tensor, "image_id": image_id}

def collate_fn(batch):
    return batch

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['data/advanced_depoisoned_model.pth'])
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    parser.add_argument('--conf-thresh', type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = setup_cfg()
    
    models = []
    for path in args.models:
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
            
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(path)
        model.to(device)
        model.eval()
        models.append(model)
        
    print(f"Loaded {len(models)} models for ensembling.")

    dataset = TestDataset('data/test_set')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    results = []
    
    print("Starting ensemble inference...")
    with torch.no_grad():
        for batch in data_loader:
            for d in batch:
                d["image"] = d["image"].to(device)
                
            batch_boxes = [[] for _ in range(len(batch))]
            batch_scores = [[] for _ in range(len(batch))]
            batch_labels = [[] for _ in range(len(batch))]
            
            for model in models:
                outputs = model(batch)
                for i, output in enumerate(outputs):
                    instances = output["instances"]
                    batch_boxes[i].append(instances.pred_boxes.tensor)
                    batch_scores[i].append(instances.scores)
                    batch_labels[i].append(instances.pred_classes)
                    
            for i, d in enumerate(batch):
                image_id = d["image_id"]
                
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
