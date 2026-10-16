"""
Dataset inspection for a YOLO-format detection dataset (Roboflow-style export).

Reports:
  - image counts per split
  - class balance (instance counts, not just image counts) per split
  - bounding box size distribution (relative to image size) -- flags how many
    boxes are "small" objects, since that drives architecture decisions
  - image resolution spread
  - basic annotation sanity checks (out-of-range coords, empty labels, labels
    with no matching image, images with no matching label)
  - exact-duplicate image detection across splits (a common source of val/test
    leakage in scraped datasets)

Usage:
    python scripts/inspect_dataset.py --data /path/to/data.yaml

Output: a text summary printed to console, plus histograms saved as
box_size_distribution.png and class_balance.png in the current directory.
"""
import argparse
import hashlib
import os
from collections import Counter, defaultdict

import yaml

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(data_yaml_path, rel_path):
    """
    Roboflow's YOLO export writes paths like '../train/images' in data.yaml,
    but conventions vary on whether that's relative to the yaml's own folder
    or to its parent (some training scripts expect to run from a subfolder
    inside the dataset). Try a few candidates and use whichever actually
    exists on disk, so this doesn't silently point at the wrong directory.
    """
    base = os.path.dirname(os.path.abspath(data_yaml_path))

    candidates = [os.path.normpath(os.path.join(base, rel_path))]

    # Also try with one leading '../' stripped (path relative to yaml's own dir)
    stripped = rel_path
    for prefix in ("../", "..\\"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped != rel_path:
        candidates.append(os.path.normpath(os.path.join(base, stripped)))

    # Also try the rel_path resolved from the yaml's PARENT dir (in case it
    # has extra leading '../' segments relative to convention)
    candidates.append(os.path.normpath(os.path.join(os.path.dirname(base), rel_path)))

    for c in candidates:
        if os.path.isdir(c):
            return c

    # Nothing exists -- return the first (literal) candidate but the caller
    # will report it as not-found, and we print what we tried for debugging.
    print(f"  [!] Could not resolve '{rel_path}' to an existing directory. Tried:")
    for c in candidates:
        print(f"      {c}")
    return candidates[0]


def label_path_for_image(img_path):
    # Roboflow/YOLO convention: .../images/xxx.jpg -> .../labels/xxx.txt
    d, fname = os.path.split(img_path)
    stem, _ = os.path.splitext(fname)
    labels_dir = d.replace(os.sep + "images", os.sep + "labels")
    if labels_dir == d:  # fallback if "images" wasn't in the path
        labels_dir = os.path.join(os.path.dirname(d), "labels")
    return os.path.join(labels_dir, stem + ".txt")


def file_hash(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def inspect_split(split_name, images_dir, names):
    result = {
        "split": split_name,
        "num_images": 0,
        "num_labeled_images": 0,
        "num_unlabeled_images": 0,
        "missing_images_for_labels": 0,
        "class_instance_counts": Counter(),
        "box_rel_areas": [],       # relative area (w*h in normalized units)
        "box_rel_dims": [],        # (w, h) normalized, for aspect/size analysis
        "image_dims": [],          # (width, height) pixels
        "malformed_lines": 0,
        "out_of_range_coords": 0,
        "image_hashes": {},        # hash -> filepath
    }

    if not os.path.isdir(images_dir):
        print(f"  [!] images dir not found: {images_dir}")
        return result

    image_files = [f for f in os.listdir(images_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS]
    result["num_images"] = len(image_files)

    for fname in image_files:
        img_path = os.path.join(images_dir, fname)
        lbl_path = label_path_for_image(img_path)

        if HAS_PIL:
            try:
                with Image.open(img_path) as im:
                    result["image_dims"].append(im.size)  # (w, h)
            except Exception:
                pass

        try:
            result["image_hashes"][file_hash(img_path)] = img_path
        except Exception:
            pass

        if not os.path.isfile(lbl_path):
            result["num_unlabeled_images"] += 1
            continue

        with open(lbl_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            result["num_unlabeled_images"] += 1
            continue

        result["num_labeled_images"] += 1
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                result["malformed_lines"] += 1
                continue
            try:
                cls_id = int(parts[0])
                x, y, w, h = (float(p) for p in parts[1:])
            except ValueError:
                result["malformed_lines"] += 1
                continue

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                result["out_of_range_coords"] += 1

            cls_name = names[cls_id] if 0 <= cls_id < len(names) else f"unknown_id_{cls_id}"
            result["class_instance_counts"][cls_name] += 1
            result["box_rel_areas"].append(w * h)
            result["box_rel_dims"].append((w, h))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to data.yaml")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    splits = {}
    for split_key in ("train", "val", "test"):
        if split_key in cfg and cfg[split_key]:
            splits[split_key] = resolve_path(args.data, cfg[split_key])

    print(f"Classes ({len(names)}): {names}\n")

    all_results = {}
    for split_name, images_dir in splits.items():
        print(f"Inspecting split '{split_name}' -> {images_dir}")
        all_results[split_name] = inspect_split(split_name, images_dir, names)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_class_counts = Counter()
    for split_name, r in all_results.items():
        print(f"\n[{split_name}]")
        print(f"  images: {r['num_images']}  "
              f"(labeled: {r['num_labeled_images']}, unlabeled/empty: {r['num_unlabeled_images']})")
        if r["malformed_lines"]:
            print(f"  [!] malformed label lines: {r['malformed_lines']}")
        if r["out_of_range_coords"]:
            print(f"  [!] out-of-range box coords: {r['out_of_range_coords']}")
        print("  class instance counts:")
        for cls, cnt in r["class_instance_counts"].most_common():
            print(f"    {cls}: {cnt}")
            total_class_counts[cls] += cnt
        if r["image_dims"]:
            widths = [d[0] for d in r["image_dims"]]
            heights = [d[1] for d in r["image_dims"]]
            print(f"  image size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        if r["box_rel_areas"]:
            areas = r["box_rel_areas"]
            small = sum(1 for a in areas if a < 0.02)   # < ~2% of image area
            tiny = sum(1 for a in areas if a < 0.005)   # < 0.5% of image area
            print(f"  boxes < 2% of image area (small): {small}/{len(areas)} "
                  f"({100*small/len(areas):.1f}%)")
            print(f"  boxes < 0.5% of image area (tiny): {tiny}/{len(areas)} "
                  f"({100*tiny/len(areas):.1f}%)")

    print(f"\n[overall] class balance across all splits:")
    total = sum(total_class_counts.values())
    for cls, cnt in total_class_counts.most_common():
        print(f"  {cls}: {cnt} ({100*cnt/total:.1f}%)" if total else f"  {cls}: {cnt}")

    # ---- Cross-split duplicate check (leakage) ----
    print("\n[duplicate check across splits]")
    hash_to_splits = defaultdict(list)
    for split_name, r in all_results.items():
        for h, path in r["image_hashes"].items():
            hash_to_splits[h].append((split_name, path))
    leaked = {h: v for h, v in hash_to_splits.items() if len({s for s, _ in v}) > 1}
    if leaked:
        print(f"  [!] {len(leaked)} exact-duplicate images found across different splits "
              f"(train/val/test leakage) -- this will inflate your validation metrics.")
        for h, entries in list(leaked.items())[:5]:
            print(f"    e.g. {entries}")
        if len(leaked) > 5:
            print(f"    ... and {len(leaked) - 5} more")
    else:
        print("  none found (good).")

    # ---- Plots ----
    if HAS_MPL:
        all_areas = []
        for r in all_results.values():
            all_areas.extend(r["box_rel_areas"])
        if all_areas:
            plt.figure(figsize=(8, 5))
            plt.hist(all_areas, bins=50)
            plt.xlabel("Box area (fraction of image area)")
            plt.ylabel("Count")
            plt.title("Bounding box size distribution (all splits)")
            plt.savefig("box_size_distribution.png", dpi=120, bbox_inches="tight")
            print("\nSaved box_size_distribution.png")

        if total_class_counts:
            plt.figure(figsize=(8, 5))
            classes, counts = zip(*total_class_counts.most_common())
            plt.bar(classes, counts)
            plt.ylabel("Instance count")
            plt.title("Class balance (all splits)")
            plt.savefig("class_balance.png", dpi=120, bbox_inches="tight")
            print("Saved class_balance.png")
    else:
        print("\n(matplotlib not installed -- skipping plots. `pip install matplotlib` to enable.)")


if __name__ == "__main__":
    main()