# Shahed Drone Detector — YOLOv9 (Ultralytics)

## Project structure
```
shahed-detector/
├── data/
│   └── data.yaml           # dataset config (copy your real one here / point --data at yours)
├── models/
│   └── yolov9c-p2.yaml     # custom arch: adds P2/4 head for small/distant objects
├── scripts/
│   ├── train.py             # training entry point
│   └── infer_realtime.py    # webcam/video/stream inference
└── runs/                    # training outputs land here (auto-created)
```

## Setup
```bash
pip install ultralytics
```
Put your actual dataset's `data.yaml` where its `train:`/`val:`/`test:` relative
paths resolve correctly (i.e. it needs to sit next to your `train/`, `valid/`,
`test/` folders, per your original summary), or just update the `--data` flag
to point at wherever it already lives.

## Baseline training
```bash
python scripts/train.py \
  --data /path/to/your/data.yaml \
  --model yolov9c.yaml \
  --weights yolov9c.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```
- `yolov9c.pt` pretrained weights (COCO) get loaded into the `yolov9c.yaml`
  architecture -- this is fine-tuning, not training from scratch, and will
  converge much faster / need less data than random init.
- Model size options (speed vs accuracy): `yolov9t` < `s` < `m` < `c` < `e`.
  For live/real-time inference on modest hardware, start with `t` or `s`.
- Watch `runs/train/.../results.png` for the loss/mAP curves once training starts.

## Modifying the architecture
`models/yolov9c-p2.yaml` is a working example: it adds a fourth detection
head at 1/4 input resolution (vs the stock 1/8, 1/16, 1/32) to better catch
small, distant targets -- likely relevant for aerial threat detection at range.
Already validated in this sandbox: 447 layers, ~24.2M params, strides
[4, 8, 16, 32].

To train with it:
```bash
python scripts/train.py --model models/yolov9c-p2.yaml --weights yolov9c.pt ...
```
Pretrained weights will still load into the unchanged layers (backbone +
existing heads); the new P2 branch starts from random init and needs the
full training run to learn.

General pattern for further edits:
1. Copy a base yaml from `ultralytics/cfg/models/v9/` (find the install path
   via `python -c "import ultralytics,os;print(os.path.dirname(ultralytics.__file__))"`).
2. Edit `backbone:`/`head:` entries -- each row is `[from, repeats, module, args]`.
3. Rebuild with `YOLO('your.yaml')` and call `.info()` to sanity-check layer
   count/params/GFLOPs before committing to a full training run.
4. Inspect visually in Netron (https://netron.app) if indices get confusing.

## Real-time inference
```bash
python scripts/infer_realtime.py --weights runs/train/shahed_v9_baseline/weights/best.pt --source 0 --show
```
`--source` accepts `0` (webcam), a video file path, an RTSP URL, or an image
directory. Use `--conf` to tune the confidence threshold once you see
real-world false-positive/negative rates -- birds vs drones at a distance
will likely need threshold tuning specific to your camera setup.

## Resources
- Ultralytics YOLOv9 docs: https://docs.ultralytics.com/models/yolov9/
- Original YOLOv9 repo: https://github.com/WongKinYiu/yolov9
- YOLOv9 paper (PGI + GELAN): arXiv 2402.13616
- Netron (architecture visualization): https://netron.app
- Roboflow blog has practical custom-dataset YOLOv9 training walkthroughs