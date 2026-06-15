#!/usr/bin/env python
"""Multi-GPU (data-parallel) wrapper around src/infer_local.py, with optional
PACKING of several model instances per GPU.

Each ~8 GB DeepSight model fits several times on an 80 GB A100, and a single bs=1
worker rarely saturates the GPU, so running N workers per GPU can raise aggregate
throughput. This launcher computes the total number of shards as
`len(gpus) * models_per_gpu`, launches one `infer_local.py` worker per shard
(pinned to its GPU via CUDA_VISIBLE_DEVICES, using the built-in --index/--num_pro
sharding), then merges the per-shard outputs — and optionally runs eval_l2.py.

Why a launcher rather than multiprocessing: each worker is a fresh process whose
CUDA_VISIBLE_DEVICES is set BEFORE torch initialises — the simplest, most robust
way to pin workers to GPUs — and it reuses the already-tested single-GPU code.

Examples
--------
    # 4 GPUs, 3 models each = 12 workers, then eval
    python src/infer_local_multi_gpu.py --gpus 0,1,2,3 --models-per-gpu 3 --eval

    # default: 1 model per detected GPU
    python src/infer_local_multi_gpu.py \
        --val_data local_data/infer_samples.jsonl --out debug/infer_results.json --eval
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER = os.path.join(REPO, "src", "infer_local.py")
EVAL = os.path.join(REPO, "src", "tools", "eval_l2.py")


def detect_gpus():
    """Physical GPU ids to use: honor CUDA_VISIBLE_DEVICES if set, else all GPUs."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    try:
        out = subprocess.check_output(["nvidia-smi", "--list-gpus"], text=True)
        return [str(i) for i, line in enumerate(out.splitlines()) if line.strip()]
    except Exception:  # noqa: BLE001
        return ["0"]


def main():
    ap = argparse.ArgumentParser(description="Data-parallel multi-GPU DeepSight inference (with GPU packing).")
    ap.add_argument("--model_path", default=os.path.join(REPO, "checkpoints", "deepsight"))
    ap.add_argument("--val_data", default=os.path.join(REPO, "local_data", "infer_samples.jsonl"))
    ap.add_argument("--out", default=os.path.join(REPO, "debug", "infer_results.json"))
    ap.add_argument("--gpus", default=None, help="comma list e.g. 0,1,2,3 (default: auto-detect / CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--models-per-gpu", type=int, default=1, dest="models_per_gpu",
                    help="model instances (workers) per GPU; total shards = gpus * this")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="seconds to wait between launching workers (smooths the load spike when packing)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--eval", action="store_true", help="run eval_l2.py on the merged output")
    ap.add_argument("--plot_dir", default="", help="passed to eval_l2.py when --eval is set")
    args = ap.parse_args()

    if not os.path.exists(args.val_data):
        sys.exit(f"val_data not found: {args.val_data}")
    if args.models_per_gpu < 1:
        sys.exit("--models-per-gpu must be >= 1")

    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else detect_gpus()
    # build the worker list: shard index -> gpu (models_per_gpu contiguous shards per GPU)
    workers = []  # (shard_idx, gpu)
    for gi, gpu in enumerate(gpus):
        for _m in range(args.models_per_gpu):
            workers.append((len(workers), gpu))
    W = len(workers)  # total shards = len(gpus) * models_per_gpu

    n_samples = sum(1 for l in open(args.val_data, encoding="utf-8") if l.strip())
    eta_min = round(n_samples * 5 / max(W, 1) / 60)
    print(f"[multi-gpu] {n_samples} samples | {len(gpus)} GPU(s) × {args.models_per_gpu} model(s) "
          f"= {W} workers | ETA ~{eta_min} min (optimistic; packing scales sub-linearly)")
    if args.models_per_gpu > 1:
        print(f"[multi-gpu] NOTE: {args.models_per_gpu} models share each GPU (~{8*args.models_per_gpu} GB "
              "of weights + activations/KV-cache); make sure that fits in GPU memory.")

    shard_dir = os.path.abspath(args.out) + ".shards"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(shard_dir, exist_ok=True)

    procs = []
    for w_i, gpu in workers:
        shard_out = os.path.join(shard_dir, f"shard_{w_i}.json")
        log_path = os.path.join(shard_dir, f"shard_{w_i}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu          # pin this worker to one physical GPU (shared if packed)
        cmd = [sys.executable, INFER,
               "--model_path", args.model_path,
               "--val_data", args.val_data,
               "--out", shard_out,
               "--index", str(w_i), "--num_pro", str(W),
               "--attn", args.attn,
               "--max_new_tokens", str(args.max_new_tokens)]
        log = open(log_path, "w")
        print(f"  worker {w_i:>2} -> GPU {gpu}  (log: {log_path})")
        procs.append((w_i, gpu, shard_out, log, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        if args.stagger > 0 and (w_i + 1) < W:
            time.sleep(args.stagger)

    print(f"[multi-gpu] running {W} workers... monitor with:  tail -f {shard_dir}/shard_0.log")

    # wait for all workers
    failed = []
    for w_i, gpu, shard_out, log, p in procs:
        rc = p.wait()
        log.close()
        print(f"  worker {w_i:>2} (GPU {gpu}): {'ok' if rc == 0 else f'FAIL(exit {rc})'}")
        if rc != 0:
            failed.append(w_i)
    if failed:
        sys.exit(f"[multi-gpu] worker(s) {failed} failed — see {shard_dir}/shard_*.log; not merging.")

    # merge shards (newline between files so the last/first lines don't fuse)
    total = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for w_i, gpu, shard_out, log, p in procs:
            if os.path.exists(shard_out):
                txt = open(shard_out, encoding="utf-8").read().strip("\n")
                if txt:
                    fout.write(txt + "\n")
                    total += txt.count("\n") + 1
    print(f"[multi-gpu] merged {total} results -> {args.out}")

    if args.eval:
        cmd = [sys.executable, EVAL, "--infer", args.out]
        if args.plot_dir:
            cmd += ["--plot_dir", args.plot_dir]
        print("[multi-gpu] eval_l2 ...")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
