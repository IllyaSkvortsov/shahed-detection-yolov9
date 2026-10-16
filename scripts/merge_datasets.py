"""
Merge N YOLO-format detection datasets into a single unified dataset with a
consistent class schema, split-preserving, with:

  - class id remapping per source (e.g. a single-class 'shahed_drone_Del' or
    '1' dataset folded into the 'shahed' class of a multi-class dataset)
  - malformed/out-of-range label line sanitization (dropped + logged)
  - filename collision handling (every copied file is prefixed with its
    source dataset tag)
  - cross-source/cross-split exact-duplicate image skipping (keeps the first
    occurrence in source order, logs the rest)
  - background (no-object) images preserved by default

Usage (repeat --tag/--yaml/--map for each source, in matching order):

    python scripts/merge_datasets.py \
      --tag dsA --yaml D:/img_detec/shahed.v5i.yolov9/data.yaml --map "" \
      --tag dsB --yaml D:/img_detec/shahed_del/data.yaml --map "shahed_drone_Del:shahed" \
      --tag dsC --yaml D:/img_detec/shahed_generic/data.yaml --map "1:shahed" \
      --output D:/img_detec/merged_dataset

--map is a comma-separated list of 'srcname:targetname' pairs. Pass "-" (empty
string) for a source whose class names already match the target schema
exactly (e.g. your original bird/not/shahed dataset needs no mapping).

--target-names defaults to the first source's class names -- override with
--target-names "bird,not,shahed" if needed.

Then point scripts/train.py --data at <output>/data.yaml.
"""
import argparse
import hashlib
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SourceDataset:
    tag: str
    yaml_path: str
    names: List[str]
    splits: Dict[str, str]
    overrides: Dict[str, str]
    id_map: Dict[int, int] = field(default_factory=dict)


def resolve_split_dir(yaml_path, rel_path):
    """Roboflow's '../' paths are ambiguous about what they're relative to --
    try multiple candidates and use whichever exists on disk."""
    base = os.path.dirname(os.path.abspath(yaml_path))
    candidates = [os.path.normpath(os.path.join(base, rel_path))]

    stripped = rel_path
    for prefix in ("../", "..\\"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped != rel_path:
        candidates.append(os.path.normpath(os.path.join(base, stripped)))

    candidates.append(os.path.normpath(os.path.join(os.path.dirname(base), rel_path)))

    for c in candidates:
        if os.path.isdir(c):
            return c

    print(f"  [!] Could not resolve '{rel_path}'. Tried: {candidates}")
    return candidates[0]


def label_path_for_image(img_path):
    d, fname = os.path.split(img_path)
    stem, _ = os.path.splitext(fname)
    labels_dir = d.replace(os.sep + "images", os.sep + "labels")
    if labels_dir == d:
        labels_dir = os.path.join(os.path.dirname(d), "labels")
    return os.path.join(labels_dir, stem + ".txt")


def load_dataset_meta(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    splits = {}
    for key in ("train", "val", "test"):
        if key in cfg and cfg[key]:
            splits[key] = resolve_split_dir(yaml_path, cfg[key])
    return names, splits


def build_id_map(source_names, target_names, overrides):
    id_map = {}
    for src_id, src_name in enumerate(source_names):
        if src_name in overrides:
            tgt_name = overrides[src_name]
        elif src_name in target_names:
            tgt_name = src_name
        else:
            raise ValueError(
                f"Class '{src_name}' (id {src_id}) has no mapping to the target "
                f"schema {target_names}. Add it to this source's --map, e.g. "
                f"'{src_name}:shahed'."
            )
        if tgt_name not in target_names:
            raise ValueError(f"Mapped target class '{tgt_name}' is not in target names {target_names}.")
        id_map[src_id] = target_names.index(tgt_name)
    return id_map


def parse_map_arg(map_str):
    """'srcname:tgtname,srcname2:tgtname2' -> {srcname: tgtname}.
    Also treats '', '-', or 'none' (any case, whitespace-insensitive) as
    "no mapping needed" -- use '-' on Windows PowerShell/cmd, where an
    actual empty string ("") passed to a native executable can get dropped
    entirely before Python ever sees it, causing a confusing argparse error."""
    if map_str is None or map_str.strip().lower() in ("", "-", "none"):
        return {}
    result = {}
    for pair in map_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        src, tgt = pair.split(":")
        result[src.strip()] = tgt.strip()
    return result


def sanitize_and_remap_label(lines, id_map):
    kept = []
    dropped = 0
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            dropped += 1
            continue
        try:
            src_cls = int(parts[0])
            x, y, w, h = (float(p) for p in parts[1:])
        except ValueError:
            dropped += 1
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            dropped += 1
            continue
        if src_cls not in id_map:
            dropped += 1
            continue
        tgt_cls = id_map[src_cls]
        kept.append(f"{tgt_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return kept, dropped


def file_hash(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def merge_split(tag, images_dir, id_map, out_images_dir, out_labels_dir,
                 seen_hashes, include_background, copy_mode, stats):
    if not os.path.isdir(images_dir):
        return
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    image_files = sorted(f for f in os.listdir(images_dir)
                          if os.path.splitext(f)[1].lower() in IMG_EXTS)

    for fname in image_files:
        src_img = os.path.join(images_dir, fname)

        try:
            h = file_hash(src_img)
        except Exception:
            stats["hash_errors"] += 1
            continue

        if h in seen_hashes:
            stats["duplicates_skipped"] += 1
            continue

        src_lbl = label_path_for_image(src_img)
        if os.path.isfile(src_lbl):
            with open(src_lbl, "r") as f:
                raw_lines = [ln.strip() for ln in f if ln.strip()]
        else:
            raw_lines = []

        kept_lines, dropped = sanitize_and_remap_label(raw_lines, id_map)
        stats["malformed_dropped"] += dropped

        if not kept_lines and not include_background:
            stats["background_skipped"] += 1
            continue

        seen_hashes.add(h)
        out_name = f"{tag}__{fname}"
        stem, _ = os.path.splitext(out_name)
        out_img_path = os.path.join(out_images_dir, out_name)
        out_lbl_path = os.path.join(out_labels_dir, stem + ".txt")

        if copy_mode == "symlink":
            if os.path.exists(out_img_path):
                os.remove(out_img_path)
            os.symlink(os.path.abspath(src_img), out_img_path)
        else:
            shutil.copy2(src_img, out_img_path)

        with open(out_lbl_path, "w") as f:
            f.write("\n".join(kept_lines))

        stats["images_written"] += 1
        for ln in kept_lines:
            cls_id = int(ln.split()[0])
            stats["class_counts"][cls_id] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append", required=True,
                         help="short unique tag for a source dataset, e.g. dsA (repeatable)")
    parser.add_argument("--yaml", action="append", required=True,
                         help="path to that source's data.yaml (repeatable, matches --tag order)")
    parser.add_argument("--map", action="append", default=None,
                         help="'srcname:tgtname,...' for that source, or '' if names already match "
                              "(repeatable, matches --tag order)")
    parser.add_argument("--target-names", type=str, default="",
                         help="comma list of final class names, default = first source's names")
    parser.add_argument("--output", type=str, required=True, help="output dataset root directory")
    parser.add_argument("--include-background", action="store_true", default=True)
    parser.add_argument("--no-include-background", dest="include_background", action="store_false")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()

    tags = args.tag
    yamls = args.yaml
    maps = args.map if args.map is not None else [""] * len(tags)

    if not (len(tags) == len(yamls) == len(maps)):
        raise ValueError(
            f"--tag ({len(tags)}), --yaml ({len(yamls)}), and --map ({len(maps)}) "
            f"counts must match -- pass one of each per source, in the same order."
        )
    if len(set(tags)) != len(tags):
        raise ValueError(f"--tag values must be unique, got: {tags}")

    sources: List[SourceDataset] = []
    for tag, yaml_path, map_str in zip(tags, yamls, maps):
        names, splits = load_dataset_meta(yaml_path)
        sources.append(SourceDataset(
            tag=tag, yaml_path=yaml_path, names=names, splits=splits,
            overrides=parse_map_arg(map_str),
        ))

    target_names = (args.target_names.split(",") if args.target_names else sources[0].names)
    target_names = [n.strip() for n in target_names]
    print(f"Target classes: {target_names}\n")

    for src in sources:
        src.id_map = build_id_map(src.names, target_names, src.overrides)
        mapping_str = ", ".join(f'{src.names[i]}({i})->{target_names[j]}({j})'
                                 for i, j in src.id_map.items())
        print(f"[{src.tag}] {src.yaml_path}")
        print(f"    class map: {{ {mapping_str} }}")

    os.makedirs(args.output, exist_ok=True)
    seen_hashes = set()
    split_dir_names = {"train": "train", "val": "valid", "test": "test"}
    grand_total = Counter()

    for split_key in ("train", "val", "test"):
        out_dir_name = split_dir_names[split_key]
        out_images_dir = os.path.join(args.output, out_dir_name, "images")
        out_labels_dir = os.path.join(args.output, out_dir_name, "labels")

        split_stats = {
            "images_written": 0, "malformed_dropped": 0, "duplicates_skipped": 0,
            "background_skipped": 0, "hash_errors": 0, "class_counts": Counter(),
        }

        print(f"\n[{split_key}]")
        for src in sources:
            if split_key not in src.splits:
                continue
            stats = {
                "images_written": 0, "malformed_dropped": 0, "duplicates_skipped": 0,
                "background_skipped": 0, "hash_errors": 0, "class_counts": Counter(),
            }
            merge_split(src.tag, src.splits[split_key], src.id_map,
                        out_images_dir, out_labels_dir, seen_hashes,
                        args.include_background, args.copy_mode, stats)
            print(f"  [{src.tag}] images written: {stats['images_written']}  "
                  f"malformed dropped: {stats['malformed_dropped']}  "
                  f"duplicates skipped: {stats['duplicates_skipped']}")
            for k in split_stats:
                if k == "class_counts":
                    split_stats[k].update(stats[k])
                else:
                    split_stats[k] += stats[k]

        print(f"  -- split total: images {split_stats['images_written']}, "
              f"class counts { {target_names[k]: v for k, v in split_stats['class_counts'].items()} }")
        grand_total.update(split_stats["class_counts"])

    out_yaml = os.path.join(args.output, "data.yaml")
    with open(out_yaml, "w") as f:
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(target_names)}\n")
        f.write(f"names: {target_names}\n")

    print(f"\nMerged dataset written to: {args.output}")
    print(f"data.yaml written to: {out_yaml}")
    final_counts: Dict[str, int] = {target_names[k]: v for k, v in grand_total.items()}
    print(f"[overall merged class balance] {final_counts}")


if __name__ == "__main__":
    main()