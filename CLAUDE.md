# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepSight is an end-to-end autonomous driving system built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). It performs long-horizon world modeling via latent BEV (Bird's-Eye-View) state prediction using a Qwen2.5-VL-3B vision-language model augmented with a DINOv3 feature extractor. In a **single forward pass** the model jointly produces three outputs: (a) latent BEV **world features** for the next 5 future frames (2 s ahead), supervised by alignment to DINOv3 features of ground-truth future BEV images; (b) an **adaptive Chain-of-Thought** text that injects external/social knowledge for long-tail scenarios; and (c) **trajectory waypoints**.

The paper (ICML 2026 submission) reports SOTA on the **closed-loop Bench2Drive** benchmark — evaluated on the official 220 short routes across 44 interactive scenarios; the five closed-loop metrics are **Driving Score (DS), Success Rate (SR), Efficiency, Comfortness, and Multi-Ability** (ablations use Route Completion / Infraction Score / DS). Open-loop L2 is reported at 0.58. Training scale (main text): **64× H20 (96 GB) GPUs, batch 128, lr 2e-5, 2 epochs**.

The accompanying paper sources are in [tex_source/](tex_source/) (`main.tex` + `sec/*.tex`). For a full paper-to-code mapping see [SRC_CODE_MAP.md](SRC_CODE_MAP.md) — consult it first when a task references a paper concept.

## Commands

### Environment
- Main training/inference env: conda **Python 3.10**, **PyTorch 2.6.0** (per README).
- Closed-loop Bench2Drive eval runs in a **separate** env with the CARLA client (`carla==0.9.16`); see the frozen `pip list` in [example.txt](example.txt).
- ⚠️ `requirements.txt` is **not** committed in this repo, so the README's `pip install -r requirements.txt` will fail — install editable and add dependencies as needed.

### Development
```bash
pip install -e .                  # install package (LLaMA-Factory fork) in editable mode
```

### Lint / Style
```bash
make quality   # check with ruff (lint + format check)
make style     # auto-fix with ruff
make license   # check license headers via tests/check_license.py
```

### Tests
```bash
make test
# equivalent: CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true pytest tests/
```

### Training
```bash
# Canonical entry point: the LLaMA-Factory CLI (src/llamafactory/cli.py → run_exp).
# It auto-switches to a torchrun launcher (src/llamafactory/launcher.py) when >1 GPU
# is visible or FORCE_TORCHRUN is set — no separate launch script needed.
llamafactory-cli train --config ./configs/ad_bev_v4.yaml

# nebula.sh wraps the same command. NOTE: its "method2" line (torchrun src/train.py)
# is dead — there is no src/train.py; the CLI handles distributed launch internally.
bash nebula.sh
```
> The training YAML `configs/ad_bev_v4.yaml` is **not** committed (no `configs/` dir). You must supply it (model path, `dataset: bench2drive_bev_train`, lr, batch size, epochs). Paper: 64× H20, bs 128, lr 2e-5, 2 epochs (main text); Appendix differs (lr 2e-4, bs 64).

### Inference
```bash
# HuggingFace transformers (debug/dev) — loads Qwen2_5_VLForConditionalGeneration,
# prefills the <|bev_token_i|> block, then model.generate(max_new_tokens=15000)
python src/infer_for_debug.py

# vLLM (production throughput) — first strip dino*/vis_head* keys, then serve.
python src/tools/merge_model_weight.py    # NOTE: src/tools/, NOT src/utils/ (README is wrong)
python scripts/vllm_infer.py              # stock LLaMA-Factory vLLM batch inferer
#   (there is no src/infer_with_vllm.py despite the README referencing it)
```

### Evaluation
```bash
# Open-loop evaluation with visualization
python src/tools/eval_and_visual.py

# Closed-loop (CARLA / Bench2Drive) — requires separate Python 3.10 env
bash bench2drive/leaderboard/scripts/run_evaluation_qwen.sh
```

## Architecture

### Component Map

| Path | Role |
|------|------|
| `src/llamafactory/` | Modified LLaMA-Factory core — training stages (SFT, DPO, KTO, PPO, RM), data loading, trainers |
| `src/transformers/models/qwen2_5_vl/` | Patched Qwen2.5-VL model with DINOv3 feature extraction added to the vision encoder |
| `src/dinov3/` | Meta's DINOv3 self-supervised ViT used as the BEV feature backbone |
| `src/llamafactory/data/ad_collator.py` | AD-specific data collator — disables token-level CE loss for BEV supervision tokens |
| `src/tools/` | Data preparation (`crop_bev_for_bench2drive.py`, `create_date_set.py`) and evaluation/visualization (`eval_and_visual.py`), weight merge (`merge_model_weight.py`) |
| `bench2drive/dataprocess/` | Upstream raw-data prep: `targetpointgen.py` (raw Bench2Drive → conversational samples), `jsonopenai.py` (Qwen3-VL API call for adaptive-CoT annotation) |
| `bench2drive/` | CARLA-based closed-loop evaluation framework (separate runtime environment); agent = `team_code/qwen_b2d_agent.py` |
| `data/dataset_info.json` | Dataset registry — maps dataset names to JSONL file paths (typically on NAS) |
| `nebula.sh` | Training launcher that dispatches to either `llamafactory-cli` or `torchrun src/train.py` |

### Data Format

Training data uses the **sharegpt** format (JSONL). Each sample includes:
- **images**: 4 historical front-camera frames + 6 surround-view frames + 5 future BEV frames (15 total)
- **messages**: user prompt (scene description, command, speed/trajectory) + assistant response

Assistant response structure:
```
<think> chain-of-thought reasoning </think>
<|start_bev_token|>[DINOv3 BEV feature tokens]<|end_bev_token|>
<answer>Future pixel tokens: [...]</answer>
<answer>Future waypoints: [(x,y), ...]</answer>
```

### Training Data Flow

```
dataset_info.json
    → ADCollator (ad_collator.py)         # removes CE loss on BEV tokens
    → Qwen2.5-VL vision encoder + DINOv3  # encodes images
    → SFT trainer (llamafactory)          # standard causal LM loss on text/waypoint tokens
```

### Key Design Decisions

- **DINOv3 integration** is injected into the Qwen2.5-VL model at `src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` (note the doubled `transformers/src/transformers` path — this is a vendored HF transformers checkout) rather than as a separate module, so loading the model loads both backbones. The HF-style DINOv3 used at runtime is `src/transformers/src/transformers/models/dinov3_vit/`; the full vendored Meta repo at `src/dinov3/` is reference only.
- **BEV supervision tokens** are masked out of the language modeling loss via `ad_collator.py`; the model learns to predict them via alignment to DINOv3 features extracted from ground-truth future RGB images. The world-loss weight is hard-coded as `loss = loss_rec + 2*loss_gen` in `modeling_qwen2_5_vl.py`.
- **World Queries** ($\mathbf{Q}_{\text{world}}$ in the paper) are not a learned tensor — they are 1305 pre-filled `<|bev_token_i|>` placeholder tokens; the LLM hidden states at those positions, projected by `vis_head`, are the predicted latent features.
- **Token fusion / output split** (see [SRC_CODE_MAP.md](SRC_CODE_MAP.md) §2.5): vision, world-query and text tokens are fused into one `inputs_embeds` sequence in `Qwen2_5_VLModel.forward` — vision features are written into image-placeholder slots via `masked_scatter` (~line 1271), while `<|bev_token_i|>`/`<|pixel_token_N|>` enter as learnable embedding-table lookups. On output (`Qwen2_5_VLForConditionalGeneration.forward`, ~line 1529), hidden states split into **only two heads**: `vis_head` (world latent, selected by `label_bev_masks`) and the shared `lm_head` (all text). **CoT and trajectory are NOT separate heads** — both are text via `lm_head`, separated only by template delimiters (`<think>…</think>` vs `<answer>…</answer>`) and the `<|pixel_token_N|>` token type, then parsed at decode time. The `<|bev_token_i|>`/`<|pixel_token_N|>` special tokens are baked into the checkpoint tokenizer (no in-repo registration; only `convert_tokens_to_ids`).
- **Two inference paths**: transformers-based (`infer_for_debug.py`) for compatibility/debugging, and vLLM-based (`infer_with_vllm.py`) requiring a weight merge step first (`merge_model_weight.py` strips `dino*`/`vis_head*` keys).
- **Bench2Drive evaluation** runs in a separate Python 3.10 environment (see `example.txt`) due to CARLA simulator dependencies.

### Paper ↔ Code Navigation

When a task references a paper concept, the authoritative mapping is in [SRC_CODE_MAP.md](SRC_CODE_MAP.md). Quick pointers:

| Paper concept | Look here |
|---------------|-----------|
| Token fusion (vision+world+text) / output→head split | `modeling_qwen2_5_vl.py` `Qwen2_5_VLModel.forward` (~1271 `masked_scatter`) and `...ForConditionalGeneration.forward` (~1529-1536); SRC_CODE_MAP.md §2.5 |
| World model / latent prediction / loss (§3.2-3.3) | `modeling_qwen2_5_vl.py` `forward` (~line 1520) |
| BEV-token masking / future-frame targets | `src/llamafactory/data/ad_collator.py` |
| World Queries `<|bev_token_i|>`, action `<|pixel_token_N|>` | `src/tools/create_date_set.py`, `infer_for_debug.py`, `qwen_b2d_agent.py` |
| Adaptive CoT `<think>…</think>` / annotation pipeline (§3.3, App. A) | `src/tools/create_date_set*.py` |
| Closed-loop inference (§3.5) | `bench2drive/team_code/qwen_b2d_agent.py` |
| Open-loop eval / visualization | `src/tools/eval_and_visual.py` |

Known gaps / stale README pointers: `configs/ad_bev_v4.yaml` and `requirements.txt` are **absent** (the `configs/` dir does not exist). The README also references three files that **do not exist** in this repo — `src/train.py`, `src/infer_with_vllm.py`, and `src/utils/merge_model_weight.py`; use `llamafactory-cli train`, `scripts/vllm_infer.py`, and `src/tools/merge_model_weight.py` respectively. All dataset/checkpoint paths are internal NAS mounts (`/mnt/nas-data-1/...`). Paper uses Qwen2.5-VL-3B (not 7B).
