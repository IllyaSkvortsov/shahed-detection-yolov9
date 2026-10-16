"""
Real-time inference: webcam, video file, RTSP stream, or a directory of images.

Usage:
    python scripts/infer_realtime.py --weights runs/train/shahed_v9_baseline/weights/best.pt --source 0
    python scripts/infer_realtime.py --weights best.pt --source path/to/video.mp4
    python scripts/infer_realtime.py --weights best.pt --source rtsp://...
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="path to trained .pt weights")
    parser.add_argument("--source", type=str, default="0",
                         help="0 for default webcam, path to video/image dir, or stream URL")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--show", action="store_true", help="display live annotated window")
    parser.add_argument("--save", action="store_true", help="save annotated output")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # stream=True keeps memory flat for long-running video/webcam sources
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show=args.show,
        save=args.save,
        stream=True,
    )

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                name = model.names[int(cls_id)]
                print(f"detected: {name}  conf={conf:.2f}")


if __name__ == "__main__":
    main()