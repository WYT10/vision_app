#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare ONNX vs NCNN on dataset splits and optional single-image latency.")
    ap.add_argument("--exe", required=True, help="Path to portable_cls_infer executable")
    ap.add_argument("--onnx-model", required=True)
    ap.add_argument("--ncnn-param", required=True)
    ap.add_argument("--ncnn-bin", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dataset-root", required=True, help="Folder containing train/ val/ test/")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", action="append", default=["val", "test"], help="Split(s) to benchmark, repeatable")
    ap.add_argument("--prep", default="crop", choices=["crop", "stretch", "letterbox"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--latency-image", default="", help="Optional single image for repeated latency test")
    ap.add_argument("--latency-repeat", type=int, default=200)
    ap.add_argument("--latency-warmup", type=int, default=20)
    args = ap.parse_args()

    exe = str(Path(args.exe).resolve())
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve()

    rows: list[dict] = []

    splits = []
    for s in args.split:
        if s not in splits:
            splits.append(s)

    backends = [
        ("onnx", ["--model", str(Path(args.onnx_model).resolve())]),
        ("ncnn", ["--model", str(Path(args.ncnn_param).resolve()), "--weights", str(Path(args.ncnn_bin).resolve())]),
    ]

    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"Skip missing split: {split_dir}")
            continue
        for backend, model_args in backends:
            summary_json = outdir / f"{backend}_{split}_summary.json"
            per_class_csv = outdir / f"{backend}_{split}_per_class.csv"
            cmd = [
                exe,
                "--backend", backend,
                *model_args,
                "--labels", str(Path(args.labels).resolve()),
                "--input", str(split_dir),
                "--prep", args.prep,
                "--size", str(args.size),
                "--threads", str(args.threads),
                "--topk", str(args.topk),
                "--eval-parent-label",
                "--quiet-per-image",
                "--summary-json", str(summary_json),
                "--per-class-csv", str(per_class_csv),
            ]
            run(cmd)
            data = read_json(summary_json)
            data["mode"] = "dataset"
            data["split"] = split
            rows.append(data)

    if args.latency_image:
        latency_image = Path(args.latency_image).resolve()
        for backend, model_args in backends:
            summary_json = outdir / f"{backend}_latency_summary.json"
            cmd = [
                exe,
                "--backend", backend,
                *model_args,
                "--labels", str(Path(args.labels).resolve()),
                "--input", str(latency_image),
                "--prep", args.prep,
                "--size", str(args.size),
                "--threads", str(args.threads),
                "--topk", str(args.topk),
                "--repeat", str(args.latency_repeat),
                "--warmup", str(args.latency_warmup),
                "--quiet-per-image",
                "--summary-json", str(summary_json),
            ]
            run(cmd)
            data = read_json(summary_json)
            data["mode"] = "latency"
            data["split"] = "single_image"
            rows.append(data)

    if not rows:
        print("No results generated.")
        return 1

    csv_path = outdir / "compare_summary.csv"
    fieldnames = [
        "mode", "backend", "split", "input_path", "preprocess", "size", "threads", "topk",
        "source_images", "repeat", "warmup", "eval_parent_label", "total", "correct", "accuracy",
        "seconds", "img_per_s", "ms_per_image",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    md_path = outdir / "compare_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Backend Comparison\n\n")
        f.write("## Dataset accuracy / throughput\n\n")
        f.write("| split | backend | accuracy | img/s | ms/img | total | correct |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            if row.get("mode") != "dataset":
                continue
            f.write(
                f"| {row.get('split','')} | {row.get('backend','')} | "
                f"{float(row.get('accuracy',0)):.4f} | {float(row.get('img_per_s',0)):.3f} | "
                f"{float(row.get('ms_per_image',0)):.3f} | {int(row.get('total',0))} | {int(row.get('correct',0))} |\n"
            )
        if any(r.get("mode") == "latency" for r in rows):
            f.write("\n## Single-image repeated latency\n\n")
            f.write("| backend | repeat | warmup | img/s | ms/img |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for row in rows:
                if row.get("mode") != "latency":
                    continue
                f.write(
                    f"| {row.get('backend','')} | {int(row.get('repeat',0))} | {int(row.get('warmup',0))} | "
                    f"{float(row.get('img_per_s',0)):.3f} | {float(row.get('ms_per_image',0)):.3f} |\n"
                )
        f.write("\n## Notes\n\n")
        f.write("- Dataset mode includes image decode + preprocessing + inference.\n")
        f.write("- Single-image latency mode isolates repeated classify() calls on one decoded image.\n")
        f.write("- Compare ONNX vs NCNN only when using the same preprocess, size, and thread count.\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
