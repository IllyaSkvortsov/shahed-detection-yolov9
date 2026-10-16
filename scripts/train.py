"""
Baseline YOLOv9 training script (Ultralytics framework).

Usage:
    python scripts/train.py --data data/data.yaml --model yolov9c.yaml --weights yolov9c.pt

Notes:
- `--model` points to the ARCHITECTURE yaml (structure/layers). Use this if you
  want to train from scratch or after editing layers.
- `--weights` (optional) loads pretrained COCO weights into that architecture
  for transfer learning -- almost always what you want to start with, even if
  you plan to modify layers later, since GELAN backbone features transfer well.
- Drop --weights entirely to train from random init (only if you've changed
  the architecture enough that pretrained weights won't load cleanly).
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/data.yaml", help="path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov9c.yaml",
                         help="architecture yaml (yolov9t/s/m/c/e.yaml) or a custom path")
    parser.add_argument("--weights", type=str, default="yolov9c.pt",
                         help="pretrained weights to fine-tune from; pass '' to train from scratch")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640,
                         help="input resolution -- bump this (e.g. 960-1280) if targets are small/far, "
                              "at the cost of speed/VRAM")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="'0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="shahed_v9_baseline")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Build model: architecture yaml, then load matching pretrained weights (transfer learning)
    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation defaults are reasonable, but for a "sky background, small
        # object" task consider tuning these once baseline works:
        # mosaic=1.0, scale=0.5, fliplr=0.5, degrees=0.0 (birds/drones don't
        # have a canonical "up" the way ground objects do, so rotation aug can help)
        plots=True,
        val=True,
    )

    # Run validation on the held-out val split and print metrics
    metrics = model.val(data=args.data, device=args.device)
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()