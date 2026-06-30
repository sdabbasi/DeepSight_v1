# Running DeepSight locally (Bench2Drive)

Command runbook for the full local pipeline: data creation → training (full-FT or LoRA) →
**open-loop** inference + L2 eval (§3) → **closed-loop** CARLA evaluation (§4) — on locally
recorded Bench2Drive samples / a local CARLA server (no NAS).

Deep explanations live elsewhere — consult those instead of duplicating here:
- Architecture / paper↔code: [CLAUDE.md](CLAUDE.md), [SRC_CODE_MAP.md](SRC_CODE_MAP.md)
- Experiment rationale / results: [RESEARCH_LOG.md](RESEARCH_LOG.md)
- Closed-loop CARLA **setup** (Vulkan / driver / install): [CARLA.md](CARLA.md)

> Env: conda **deepsight** (Python 3.10, torch 2.6, transformers 4.56.x).

---

## 0. One-time setup

```bash
conda activate deepsight

# released checkpoint (~8.8 GB)
hf download zhangthu/deepsight --local-dir ./checkpoints/deepsight

# patch installed transformers (DINOv3 world head) + repoint the checkpoint's dead
# dinov3_config NAS path. (--revert restores stock.)
python scripts/setup_local_inference.py --ckpt ./checkpoints/deepsight

# only needed for MULTI-GPU full-FT training (ZeRO): pin deepspeed
pip install 'deepspeed==0.16.9'
```

---

## 1. Data creation (sharegpt JSONL)

**Inference data** — from raw Bench2Drive scenes (one `.tar.gz` per scene, real pixels):
```bash
hf download rethinklab/Bench2Drive AccidentTwoWays_Town12_Route1102_Weather10.tar.gz \
    --repo-type dataset --local-dir ./local_data/bench2drive_raw
tar xzf ./local_data/bench2drive_raw/AccidentTwoWays_Town12_Route1102_Weather10.tar.gz \
    -C ./local_data/bench2drive_raw/
python src/tools/build_local_infer_jsonl.py \
    --scenes local_data/bench2drive_raw/AccidentTwoWays_Town12_Route1102_Weather10 \
    --out local_data/infer_samples.jsonl --stride 10 --limit 30   # --limit 0 = all frames
```

**Training data** — from prepared base scenes (15-img samples; `--stride` = samples/scene):
```bash
python src/tools/build_local_train_jsonl.py \
    --scenes $(head -6 local_data/ready_scenes.txt) \
    --out local_data/train_samples.jsonl --stride 8
# for an un-cropped scene first: python src/tools/crop_bev_for_bench2drive_local.py <scene_dir> ...
```

> Each training config reads its **own** `dataset_dir` under `local_data/` (a one-entry
> `dataset_info.json`); the repo's `data/dataset_info.json` is left untouched. Dataset name is
> always `bench2drive_bev_train` by convention — the AD pipeline wraps every SFT run regardless.

---

## 2. Training — `scripts/train.sh <config> [overrides] [--test <test.jsonl>]`

Thin wrapper: timestamped run dir + `run.log` + 3-loss plot + regime-aware `--test`.
`CUDA_VISIBLE_DEVICES` picks GPUs; `FORCE_TORCHRUN=1` triggers the distributed launcher;
trailing `key=value` args override the YAML.

```bash
# smoke (1 GPU, 1 step) — validate the path
CUDA_VISIBLE_DEVICES=0 scripts/train.sh configs/ad_bev_train_smoke.yaml

# full-FT, N GPUs — MUST add ZeRO-2 (plain DDP OOMs: ~30 GB AdamW state replicated/GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 \
    scripts/train.sh configs/ad_bev_train_local.yaml deepspeed=examples/deepspeed/ds_z2_config.json

# LoRA, 1 GPU (no ZeRO needed), with post-training open-loop L2 on the test set
CUDA_VISIBLE_DEVICES=0 scripts/train.sh \
    configs/ad_bev_e2_3_LORA_lambda2_seed0.yaml \
    --test local_data/e2_lora/test.jsonl
```

Key rules:
- **full-FT multi-GPU → ZeRO required** (`ds_z2_config.json`; too big → `ds_z3` / `*_offload`).
  **LoRA → no ZeRO** (frozen trunk, tiny optimizer state) — for N GPUs just add the devices +
  `FORCE_TORCHRUN=1` (plain DDP). Keep the **same #GPUs across arms you compare** (effective batch
  = `per_device_bs × #GPUs × grad_accum`).
- **Sets (train / eval / test):** the YAML's `dataset` is the **train** split and `eval_dataset`
  the **eval** split (monitored as `eval_loss` / early-stopping *during* training — **not** the test
  set); **`--test <test.jsonl>`** runs open-loop L2 on the untouched **test** split *after* training
  (use the registry's `test.jsonl`, e.g. `local_data/e2_lora/test.jsonl`).
- **`world_loss_weight: λ`** (YAML) is the world-loss knob: `loss = loss_rec + λ·loss_gen`
  (default 2.0; `0.0` ablates the world objective).
- `cutoff_len: 10000` must stay > sequence length so the 1305-token BEV block isn't truncated.

Output: `saves/<exp>/<unixtime>_<exp>/` with `run.log`, `trainer_log.jsonl`, `training_loss.png`,
`losses_split.png` (total / `loss_rec` / `loss_gen`), and **one final model** (or a LoRA adapter +
`merged/` if `--test` ran). `save_strategy: "no"` → no `checkpoint-*/` dirs. `saves/` is gitignored.

---

## 3. Inference + open-loop L2 eval

```bash
# single GPU
CUDA_VISIBLE_DEVICES=0 python src/infer_local.py \
    --model_path checkpoints/deepsight \
    --val_data local_data/infer_samples.jsonl \
    --out debug/infer_results.json --attn sdpa
python src/tools/eval_l2.py --infer debug/infer_results.json --plot_dir debug/traj_plots

# multi-GPU (auto-shards, merges, optionally evals)
python src/infer_local_multi_gpu.py \
    --model_path checkpoints/deepsight \
    --val_data local_data/infer_samples.jsonl \
    --out debug/infer_results.json \
    --gpus 0,1,2,3 --models-per-gpu 3 --stagger 3 --attn sdpa --test --plot_dir debug/traj_plots
```

Multi-GPU flags: `--gpus` (auto-detects; honors `CUDA_VISIBLE_DEVICES`), **`--models-per-gpu N`**
(pack N ~8 GB copies/GPU; total shards = `gpus × N`), **`--stagger S`** (seconds between worker
launches — spreads the one-time weight loads to avoid host RAM/disk spikes). Per-shard
outputs/logs under `<out>.shards/`; a failed shard is reported and merge skipped (re-run that shard).

**Evaluate a trained run** instead of the released ckpt — reuse the commands above, only changing
`--model_path` (and `--val_data <test.jsonl>`):
- full-FT → the run dir is a full checkpoint: `--model_path saves/<exp>/<run_dir>`
- LoRA → the run dir is only an adapter; merge it once, then use `--model_path <run_dir>/merged`:
  ```bash
  CUDA_VISIBLE_DEVICES=0 llamafactory-cli export <run_dir>/export_config.yaml   # → <run_dir>/merged
  ```
  (`scripts/train.sh … --test <test.jsonl>` already does this merge-then-eval automatically at train time.)

### Reproducing the paper's open-loop L2 (≈ 0.58)
Favor scenario diversity over frame density (consecutive frames are near-duplicates): build one big,
coarse-stride test set, then **rerun the multi-GPU infer above** with `--test` (its defaults already
point at `local_data/infer_samples.jsonl` + `checkpoints/deepsight`):
```bash
python src/tools/build_local_infer_jsonl.py --scenes $(cat local_data/ready_scenes.txt) \
    --out local_data/infer_samples.jsonl --stride 25 --limit 0       # ~8k samples (≈ all 1000 scenes)
```
Caveats: DeepSight predicts to 2 s → `eval_l2.py` reports 1 s/2 s only; `bench2drive_base` likely
overlaps training data so L2 may read lower than the reported 0.58. Closed-loop DS/SR needs CARLA.

---

## 4. Closed-loop CARLA evaluation (DS / SR)

Drives the model through Bench2Drive routes in the CARLA simulator and reports the closed-loop
metrics. **Results are written into the model's own exp dir**, under a `closed_loop/` folder:
`saves/<exp>/<run_dir>/closed_loop/<routes>.{json,log}` (+ `<routes>_summary.txt`, `<routes>_viz/`).

> Prereq: a working CARLA + Vulkan stack and the non-root `carla` user — full setup in
> [CARLA.md](CARLA.md). **Run all three scripts as root** (they drop to `carla`; UE4 refuses root):
> `run_carla_closed_loop.sh` also creates the `closed_loop/` dir writable by `carla` (saves/ run
> dirs are root-owned). The eval has a cleanup trap, so Ctrl-C / `kill` tears down its python — and,
> in `CARLA_NO_LAUNCH=1` mode, leaves the externally-managed server running.

**Recommended: server and eval as separate processes** (so killing the eval never orphans CARLA,
and you skip the ~60 s CARLA boot on every retry). Start the server once, then run the eval against
it with `CARLA_NO_LAUNCH=1`:
```bash
# 1) start the CARLA server once (own terminal). Boots on GPU 3, port 30000.
GPU=3 PORT=30000 bash scripts/start_carla.sh

# 2) run the eval against it — always pass CKPT (checkpoint), ROUTES (data), GPU; repeat freely
CARLA_NO_LAUNCH=1 GPU=3 PORT=30000 \
CKPT=saves/<exp>/<run_dir> \
ROUTES=leaderboard/data/smoke_test_town03.xml \
    bash scripts/run_carla_closed_loop.sh

# 3) stop the server when done
GPU=3 bash scripts/stop_carla.sh
```

**All-in-one** (the eval launches + tears down its own CARLA — simplest for a single run):
```bash
CKPT=saves/<exp>/<run_dir> \
ROUTES=leaderboard/data/smoke_test_town03.xml \
GPU=3 \
    bash scripts/run_carla_closed_loop.sh
```

- **`CKPT`** — model dir = a saves run-dir root (e.g. `saves/deepsight/original_deepsight` for the
  released model, or any trained run). A `checkpoint-XXXX` subdir attaches `closed_loop/` to its parent.
- **`ROUTES`** — the route XML to drive: `leaderboard/data/smoke_test_town03.xml` (1-route smoke) or
  `leaderboard/data/bench2drive220.xml` (full 220 — long; needs AdditionalMaps for Town06+/11+).
- **`GPU`** — a **free** Vulkan/CUDA index (shared node; Vulkan adapter = CUDA index). `PORT`/`TM_PORT`
  also overridable for parallel runs.

Outputs → `<run_dir>/closed_loop/`: per-route scores `<routes>.json`, full log `<routes>.log`,
agent visualizations `<routes>_viz/`, DS/SR summary `<routes>_summary.txt`.

**Read the scores** of any run (aggregates per-route `records[]`, since the leaderboard's
`global_record` is often empty on single-route / resume runs):
```bash
python scripts/carla_score.py saves/<exp>/<run_dir>/closed_loop/<routes>.json
#   → Driving Score (DS), Success Rate (SR), Route Completion
```

> CARLA is crash-prone; the evaluator runs with `--resume=True`, so re-running the **same** command
> continues an interrupted route set. Kill stuck servers with `bench2drive/tools/clean_carla.sh`.

---

## 5. Revert the transformers overlay

```bash
python scripts/setup_local_inference.py --revert
```
