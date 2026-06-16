**Project:** DeepSight

> Companion docs: [CLAUDE.md](CLAUDE.md) (repo guide), [SRC_CODE_MAP.md](SRC_CODE_MAP.md)
> (paper↔code), [INPUT_FORMAT.md](INPUT_FORMAT.md) (token formats),
> [WORLD_MODEL_JEPA.md](WORLD_MODEL_JEPA.md) (world-model critique → JEPA),
> [RUN_LOCAL_INFERENCE.md](RUN_LOCAL_INFERENCE.md) / [RUN_LOCAL_TRAINING.md](RUN_LOCAL_TRAINING.md)
> (runbooks).

---

## Current Direction

DeepSight (paper title: *Long-Horizon World Modeling via Latent States Prediction for
End-to-End Autonomous Driving*, ICML 2026 submission) is a **Qwen2.5-VL-3B + frozen
DINOv3** driving VLM built on a **LLaMA-Factory** fork. It ships as a **research dump
wired to an internal cluster** — NAS data paths, an incomplete vendored `transformers`,
missing `configs/`/`requirements.txt`, stale README pointers, and a CARLA-only evaluation
loop. The near-term goal is to make the repo **fully runnable locally without CARLA** —
inference, open-loop L2 eval, and training — on locally recorded Bench2Drive samples,
fixing every bug along the way. The medium-term goal (groundwork laid, not yet
implemented) is to replace the world-model's two weak design choices — the **frozen
external DINOv3 target** and the **god-eye top-down BEV source** — with a **temporal
JEPA** (an EMA in-domain BEV encoder on the future frames); see
[WORLD_MODEL_JEPA.md](WORLD_MODEL_JEPA.md).

**Discipline:** upstream files are left byte-for-byte unchanged; every modification lives
in a renamed `*_local` copy or a new file. The sole exception is **one line** in
`data/dataset_info.json` (a dataset `file_name` repoint), made by explicit request.

### Status to date

| # | item | result |
|---|---|---|
| S1 | Released checkpoint loads + single-scene inference | L2 1s ≈ 0.043 m / 2s ≈ 0.066 m (zero weight mismatches) |
| S2 | `transformers` import / DINOv3 head wiring | exec-shim onto installed transformers; repo file is live source, breakpoints bind |
| S3 | Local Bench2Drive data pipeline (no NAS, no CARLA) | sharegpt JSONL built from `rethinklab/Bench2Drive` `.tar.gz` scenes |
| S4 | Multi-GPU inference (+ GPU packing) | `gpus × models_per_gpu` workers, shard→merge→eval; verified 2 GPU & 2×2 packed |
| S5 | Training path (DINOv3-supervised, current style) | smoke 1-step + 4-GPU DeepSpeed ZeRO-2 step verified; `loss = loss_rec + 2·loss_gen` |
| S6 | Random-init "from scratch" checkpoint | `loss_rec ≈ 12.5 ≈ ln(vocab)` confirms truly random init |
| S7 | World-model design critique (JEPA / BEV) | documented; defines the two swap seams for the JEPA upgrade |

---

## Background — the system in depth

### 1. What the model is and what it produces

DeepSight is a **unified generative-understanding VLM** ($M_{\text{uni}}$ in the paper):
from multi-view + historical camera frames it produces, in a **single forward pass**,
three outputs:

- **(a) Latent BEV world features** $\mathbf{F}=[f_0..f_4]$ for the next **5 future frames
  (2 s ahead)** — supervised by alignment to **DINOv3** features of ground-truth future
  BEV images via an MSE "world loss";
- **(b) An adaptive Chain-of-Thought** $T_{\text{cot}}$ (`<think>…</think>`) that injects
  external/social knowledge for long-tail scenarios (placeholder `<think> None </think>`
  = $T_{\text{cot}}^{\emptyset}$ when no reasoning is needed);
- **(c) Trajectory waypoints** $\mathbf{P}_t$.

The single most important file is the patched model:
[modeling_qwen2_5_vl.py](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py).
Architecture facts (verified): hidden=2048, 36 layers, 16 attn heads, 2 KV heads (GQA),
head_dim=128, tie_word_embeddings=True, mRoPE [16,24,24], vocab=153536. DINOv3 is
**DINOv3-ViT-L/16** (~300M params, hidden **1024**, patch 16, pretrained LVD-1689M),
held frozen (`requires_grad_(False)`); `vis_head = nn.Linear(2048 → 1024, bias=False)`
maps LLM hidden states into DINOv3 space.

### 2. The core mechanism in code (token fusion → two-head split)

**Input side — one fused `inputs_embeds` sequence.** Three token streams coexist in the
same sequence (`Qwen2_5_VLModel.forward`, SRC_CODE_MAP §2.5):

| Stream | IDs | How its embedding is produced |
|---|---|---|
| **Text** (prompt, CoT, waypoint text) | normal vocab | embedding-table lookup |
| **World queries** `<\|bev_token_i\|>`, `<\|start/end_bev_token\|>`, action `<\|pixel_token_N\|>` | added-vocab special IDs (baked into the checkpoint tokenizer; repo only `convert_tokens_to_ids`) | **learnable** embedding-table lookup — this *is* $\mathbf{Q}_{\text{world}}$ |
| **Vision** (4 history + 6 surround frames) | repeated `image_token_id` placeholders | run through Qwen ViT, then `masked_scatter`'d into the placeholder slots (~L1271) |

Line 1271's `masked_scatter` is the fusion point: vision features land in image slots,
world-query/pixel tokens enter as learnable embeddings, text as table lookups — all in
template order. Self-attention over this fused sequence is the paper's "deep
self-attention" where $\mathcal{X}$ and $\mathbf{Q}_{\text{world}}$ interact (§3.5).

**Output side — only TWO `nn.Module` heads** (`...ForConditionalGeneration.forward`,
~L1529-1544):
- `vis_head` — fed **only** the `<|bev_token_i|>` positions (selected by the
  `label_bev_masks` boolean), → 1024-d latent $\mathbf{F}$ (world model);
- shared `lm_head` — over the whole sequence (all text).

**CoT and trajectory are NOT separate heads** — both are text through `lm_head`,
distinguished only by (i) token type (waypoints use the dedicated `<|pixel_token_N|>`
vocab; CoT uses ordinary tokens), (ii) template delimiters (`<think>…</think>` vs two
`<answer>…</answer>` blocks), and (iii) decode-time regex parsing. The only
architecturally separate output path is the world latent (`vis_head`).

**World Queries are not a learned tensor** — they are **1305 pre-filled `<|bev_token_i|>`
placeholder tokens**: `5 frames × (256 patches + 1 CLS + 4 register) = 5 × 261 = 1305`
(256×256 image, patch 16 → 16×16=256 patches). The LLM hidden states at those positions,
projected by `vis_head`, *are* the predicted latent. Because they're prefilled (a prefix),
the paper's "parallel prediction in a single pass" is realized as the prefill stage; the
model then autoregressively emits CoT + waypoints after `<|end_bev_token|>`.

**Loss** (`loss = loss_rec + 2*loss_gen`): `loss_rec` = CE over text ($L_{\text{traj}} +
L_{\text{cot}}$ lumped, both plain text), `loss_gen` = `MSE(vis_embeds, DINOv3(future_BEV))`.
The AD collator ([ad_collator.py](src/llamafactory/data/ad_collator.py)) pops the **last 5
images** as BEV targets (resized 256×256 → `pixel_values_bevs`), sets the BEV span's labels
to `IGNORE_INDEX` (so CE doesn't apply to BEV tokens), and builds `label_bev_masks` /
`bevs_masks` / `template_mask` (the last drops the 4 register tokens per frame from
supervision).

### 3. Data format and pipeline

Training uses **sharegpt** JSONL, **15 images/sample** = 4 historical CAM_FRONT (at
−2.0/−1.5/−1.0/−0.5 s) + 6 surround current frames + **5 future BEV frames** (the DINOv3
targets, popped by the collator). The assistant response:

```
<think> {cot} </think>
<|start_bev_token|>{1305 bev tokens}<|end_bev_token|>
<answer> future pixel tokens: {…} </answer>
<answer> future waypoints: {(x,y),…} </answer>
```

The prompt carries 10 `<image>` + **target pixel tokens** (route goal projected to BEV
pixels) + historical trajectory (metric meters) + speed + a `<CoT_flag_*>` toggle —
*not* a "Mission Goal" string. Verified token IDs: `<|image_pad|>`=151655, bev
151671–152975, pixel 152976–153486; each `<image>`→299 `<|image_pad|>` via the processor;
a sample is ≈ 4540 tokens (≈2990 image + 1305 bev + ~245 text). The future BEV crops are
`rgb_bev_{0,5,10,15,20}th-hz` (512×512 crops of CARLA's `rgb_top_down`; the 4 future
frames are ego-motion-compensated). Upstream prep: `targetpointgen.py` (raw → samples),
`crop_bev_for_bench2drive.py` (BEV targets), `create_date_set.py` (builder),
`jsonopenai.py` (Qwen3-VL CoT annotation).

### 4. Paper scale, results, and paper↔code discrepancies

- **Reported results:** SOTA on **closed-loop Bench2Drive** (official **220 short routes /
  44 interactive scenarios**); five metrics: **DS, SR, Efficiency, Comfortness,
  Multi-Ability** (ablations use Route Completion / Infraction Score / DS). **Open-loop
  L2 = 0.58.**
- **Training scale:** **64× H20 (96 GB)**, **batch 128, lr 2e-5, 2 epochs** (main text);
  Appendix differs (**lr 2e-4, batch 64**, frozen vision tower).
- **Discrepancies worth knowing:** (i) the world-loss weight is **hard-coded to 2**
  (`loss_rec + 2*loss_gen`), but the paper's $\lambda_{\text{world}}$ sensitivity table
  reports **best = 1.0**; (ii) base model is **Qwen2.5-VL-3B** (an old note said 7B —
  wrong); (iii) `merge_model_weight.py` strips `dino*`/`vis_head*` for vLLM serving,
  confirming the **world head is training-only** machinery.
- **Stale/missing in-repo:** `configs/ad_bev_v4.yaml` and `requirements.txt` are absent;
  README references `src/train.py`, `src/infer_with_vllm.py`,
  `src/utils/merge_model_weight.py` which **do not exist** (use `llamafactory-cli train`,
  `scripts/vllm_infer.py`, `src/tools/merge_model_weight.py`). All dataset/checkpoint
  paths are internal NAS mounts.

**Key takeaway that drives this project:** the world head shapes representations during
training but is **never read at inference** — the deployed model is camera-only and emits
waypoints through `lm_head`. That, plus the privileged top-down BEV target, is exactly
what the JEPA redesign targets.

---

## Cumulative Progress

### Enablement — inference / eval (no CARLA)

- **transformers / DINOv3 wiring.** The vendored `src/transformers/` tree is incomplete and
  unimportable; installed transformers ≥4.56 already ships `models/dinov3_vit`, so the only
  genuinely-patched file is `modeling_qwen2_5_vl.py`. `scripts/setup_local_inference.py`
  installs a **shim** into site-packages that `exec()`s the repo's modeling file compiled
  with the repo path as filename — so the repo file is the live source (edits + debugger
  breakpoints bind) while a declared `__all__` keeps `define_import_structure` exposing the
  classes for top-level imports. `--revert` restores the `.orig`.
- **Dead `dinov3_config` path.** The checkpoint's `config.json` pointed at a NAS path; the
  setup script regenerates the correct DINOv3-ViT-L/16 config locally and repoints it.
- **No usable dataset.** The ModelScope dataset is a 65 GB text-only JSONL with dead NAS
  image paths. Pulled real scenes from official `rethinklab/Bench2Drive` (one `.tar.gz` per
  scene) and built the sharegpt JSONL with `src/tools/build_local_infer_jsonl.py`, reusing
  the prompt/answer/projection math from `bench2drive/dataprocess/targetpointgen.py`.
  Inference consumes only the 10 input cameras; the 5 future-BEV slots use a placeholder
  (training-only targets).
- **Prefill format.** The checkpoint expects the goal as **target pixel tokens** + a
  `<CoT_flag_*>` toggle, and the assistant is **BEV-first** (`<|start_bev_token|>…` then
  `<think>` then `<answer>` blocks). Fixed `add_bev_text` in `src/infer_local.py` (copy of
  `infer_for_debug.py`) to prefill BEV-first; `<CoT_flag_False>` since no local CoT annos.
- **Eval.** `src/tools/eval_and_visual_local.py` (copy) fixes a 5-vs-4 unpack bug in
  `main_for_eval_l2`; `src/tools/eval_l2.py` computes 1s/2s L2 directly (DeepSight predicts
  only to 2 s = 4 waypoints) with per-scene breakdown + plots.
- **Multi-GPU + packing.** `src/infer_local_multi_gpu.py` launches `len(gpus) ×
  models_per_gpu` workers (each pinned via `CUDA_VISIBLE_DEVICES`, using `infer_local.py`'s
  `--index/--num_pro` sharding), `--stagger` smooths the load spike, then **waits for all
  shards** (blocking `p.wait()` loop), merges, and optionally runs `eval_l2.py` once on the
  merged file. A failed shard aborts the merge so it can be re-run.

### Enablement — training (current DINOv3-supervised style)

- **Missing modules.** The repo's `road_collator.py` imports `utils.obj_utils` /
  `vis_utils` / `cls_utils`, which are absent — every entry point failed to import. Added
  importable **stubs** (`RoadCollector` is unused by the Bench2Drive AD pipeline).
- **Dataset registry.** Consolidated to the single hardcoded `data/dataset_info.json`
  (the only filename LLaMA-Factory reads, `DATA_CONFIG`); repointed
  `bench2drive_bev_train.file_name` to `local_data/train_samples.jsonl` (one-line edit,
  user-approved) so `dataset_dir: data` works. `src/tools/build_local_train_jsonl.py`
  builds the **15-image** training sample (10 input + 5 real BEV crops, absolute paths).
- **Collator behavior confirmed.** The fork's `get_dataset` defers preprocessing;
  `ad_collator.py` pops the last 5 BEV images **before** tokenizing — so a 15-image sample
  with 10 `<image>` tags passes the `len(images)==#<image>` check — resizes them to 256×256
  and feeds the frozen DINOv3 as targets, asserting the BEV block is exactly
  `5×(256+1+4)=1305` tokens (→ `cutoff_len: 10000` to avoid truncation tripping that assert).
- **DeepSpeed pin.** transformers 4.56 requires `deepspeed<=0.16.9`; env shipped 0.19.0
  (every rank aborted at import). Fixed with `pip install 'deepspeed==0.16.9'`. 4-GPU
  ZeRO-2 step verified (cross-GPU grad sync OK).
- **Configs.** `ad_bev_train_smoke.yaml` (1 GPU, `max_steps=1`) and
  `ad_bev_train_local.yaml` (multi-GPU, ZeRO-2, 2 epochs). Runbook: `RUN_LOCAL_TRAINING.md`.

### Conceptual analysis

- **Input format** fully traced (`INPUT_FORMAT.md`): token IDs, prompt layout, and the
  ≈4540-token sample anatomy above.
- **World-model critique → temporal JEPA** (`WORLD_MODEL_JEPA.md`): see the next section.

---

## Conceptual analysis — the world model as a temporal JEPA

### JEPA in one page

**JEPA = Joint-Embedding Predictive Architecture:** predict the **embedding** of the
held-out part of the data, not its pixels. Four pieces: a **target encoder** `f_tgt` (EMA
copy of the context encoder, stop-gradient) on the held-out part; a **context encoder**
`f_ctx` on the visible part; a **predictor** `g` that consumes the *context
representation* (not raw pixels) + query/position info; and an **embedding-space loss**
(MSE/cosine). Two generalizations make it apply here: **(a) the mask can be temporal** —
the held-out region is *the future*; **(b) the predictor always consumes the context
encoder's output** — so "the predictor works on hidden states" is the definition, not a
contradiction.

### DeepSight's world head *is* a (degenerate) JEPA

| JEPA piece | DeepSight world model (temporal) |
|---|---|
| Held-out / "masked" region | the **future BEV frames** (next 5) — never fed to the VLM |
| Target encoder `f_tgt` (EMA, stop-grad) | currently a **frozen DINOv3** on the future frames (JEPA version: an EMA BEV encoder) |
| Context encoder `f_ctx` | the **VLM** (Qwen) encoding current+history cams, route, speed |
| Query / mask tokens | the **`<\|bev_token_i\|>`** world queries |
| Predictor `g` | the **LLM layers on bev-token positions + `vis_head`** |
| Loss (embedding space) | `MSE(vis_embeds, future-BEV latents)` |

So today's design is JEPA-*shaped* but uses a **fixed, external** target encoder — a
**degenerate JEPA** whose teacher never adapts to the domain.

### The two weak links and their replacement

**4.1 — Frozen external DINOv3 target.** The common worry ("DINOv3 isn't aligned with
Qwen's vision encoder") is mostly a misread — `vis_head` is a learned adapter and the
target is the VLM's *output*, not Qwen's encoder, so two encoders never need a shared
space. The *real* issues: (i) **domain/task mismatch** — DINOv3 is trained on natural web
images and is OOD on rasterized top-down BEV, not specialized for drivable space / lane
topology / agent kinematics / occupancy; (ii) a **fixed teacher = degenerate JEPA**, and
MSE-to-frozen-features can be dominated by a few high-variance channels / admit partial
collapse. **Replacement:** make the target an **EMA copy of an in-domain BEV encoder**
(stop-grad) — removes the external dependency *and* any cross-encoder mismatch by shared
lineage. Cost: a co-evolving teacher can collapse, so add **EMA + stop-grad +
predictor/asymmetry and/or VICReg/iBOT variance-covariance regularization** (the one thing
the frozen design got for free).

**4.2 — God-eye top-down BEV source.** BEV *as a representation* is the industry-standard
choice (BEVFormer/LSS/UniAD/VAD); the unrealistic part is the **source** — CARLA's
`TOP_DOWN` sensor is a clean overhead render with **no real-car analogue**, so the target
is sim-only, the pipeline carries a **privileged-information / sim-to-real gap**, and
because both training and the closed-loop benchmark are CARLA, that gap is never tested.
**Replacement:** build the future-BEV target from **onboard surround cameras (+lidar) via
an LSS/BEVFormer perception model** — producible on real datasets (nuScenes/Waymo),
task-grounded, and a well-established **privileged lidar→camera distillation** (lidar
training-only, student camera-only). Even stronger/verifiable: target **BEV
occupancy/flow** (OccWorld/UniAD-style) instead of latent features.

### The unified upgrade

| Aspect | DeepSight today | Upgraded (temporal JEPA) |
|---|---|---|
| Target encoder | frozen **DINOv3** (external, generic) | **EMA BEV encoder** (in-domain, self-distilled) |
| Target *source* | **god-eye top-down RGB** (CARLA, privileged) | **onboard surround cams + lidar** via LSS/BEVFormer (or occupancy) |
| Domain/task fit | natural-image features, OOD on BEV | driving-specific (lanes/agents/occupancy) |
| Cross-encoder gap | bridged only by `vis_head` | none — shared encoder lineage |
| Real-data ready | sim-only target | yes (lidar training-only, camera-only at test) |
| Collapse risk | none (fixed teacher) | must add EMA + stop-grad + variance/predictor reg |

Caveats to budget for: collapse prevention becomes *your* problem; an early teacher injects
perception noise (warm-start the BEV encoder with occupancy/seg supervision); matching a 3B
VLM's hidden states to a *moving* EMA target needs careful momentum/loss-weight/warmup
tuning; decide **features vs occupancy** as the target.

**Why it matters for the VLA:** the world head is an auxiliary self-supervised objective —
forcing the VLM to predict the future latent state makes its internal representations
**dynamics-aware** (the policy "imagines" consequences). It is latent forecasting: **no
pixels generated at inference**, world head is training-only, so a standard merged,
camera-only, waypoint-emitting model still serves. The JEPA swap keeps that benefit while
making the training signal **honest about the real world**.

### The two code seams the JEPA swap touches

- **Target + loss:** the DINOv3 call and `loss = loss_rec + 2*loss_gen` in
  [modeling_qwen2_5_vl.py:~1524-1544](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1524)
  — where a frozen DINOv3 becomes an EMA in-domain encoder (+ anti-collapse reg).
- **Target source:** the BEV-target prep / 1305-token machinery in
  [ad_collator.py:284-318](src/llamafactory/data/ad_collator.py#L284) — where the top-down
  render is replaced by an onboard surround-cam BEV (or occupancy) target.

---

## Planned Experiments — is the original recipe effectively trainable? (E1/E2/E3)

**Status: designed, NOT yet implemented.** Build/run each only on explicit request; this
section is the spec to follow at that point.

**Motivation.** Before swapping in JEPA we want to know whether the *current*
DINOv3-supervised recipe trains effectively — but Bench2Drive-full is far too large/slow to
answer that with a full training. The trick: this is not one expensive question
("reproduce the paper") but **three small, controlled ones**, each answerable on a tiny
slice of data:
1. **Plumbing / capacity** — can gradients drive *both* losses down at all? (E1)
2. **World-objective efficacy** — does the world loss actually *help the action output*, and does it even *learn*? (E2, E3)
3. These directly de-risk JEPA: JEPA reuses the same `<|bev_token_i|>` positions / collator
   masks / `vis_head` path, so a wiring bug found in E1 would also break JEPA; and E2/E3 tell
   us whether JEPA is *"fix a broken component"* or *"replace a useless one."*

**Shared infrastructure (build once, used by all three).**
- *Tiny / small subset builders* — reuse [build_local_train_jsonl.py](src/tools/build_local_train_jsonl.py)
  with `--limit` / a chosen scene list. E1 wants **N = 8–64 fixed** samples; E2/E3 want a
  **few hundred–few thousand** diverse samples (one-per-scenario-family, coarse stride —
  same diversity-over-density logic as the stride-25 eval).
- *Held-out split* — a scene list **disjoint** from the training subset (and, ideally,
  verified disjoint from the checkpoint's training data) so E2's eval is not leaked. New file
  e.g. `local_data/heldout_scenes.txt`; eval JSONL via `build_local_infer_jsonl.py`.
- *Eval* — existing [eval_l2.py](src/tools/eval_l2.py) (1s/2s open-loop L2) + the multi-GPU
  inferer. (The L2 is not yet paper-matched; for E2 we only need the *relative* λ=0 vs λ=2
  comparison, so the convention need not match the paper.)
- *A `λ_world` knob* — **IMPLEMENTED 2026-06-15 as a config-driven YAML arg** (`world_loss_weight`).
  The weight was hard-coded `loss = loss_rec + 2*loss_gen`; it is now
  `loss = loss_rec + getattr(self.config, "world_loss_weight", 2.0)*loss_gen`. Three minimal
  comment-and-add edits (default 2.0 → original behavior preserved): the field on
  `FinetuningArguments`, the wiring `model.config.world_loss_weight = finetuning_args.world_loss_weight`
  in `train/sft/workflow.py`, and the read in `modeling_qwen2_5_vl.py`. Set it in any training
  YAML (`world_loss_weight: 0`/`2`). The earlier env-var/shim idea was rejected (too implicit).

**Shared caveat — choice of start checkpoint.** Fine-tuning *from the released (converged)
checkpoint* makes any "improvement" nearly invisible (it's already near-optimal on
in-distribution frames). Options, per experiment: (a) **E1** — start from the released ckpt
(fastest path to ~0, pure plumbing test); (b) **E2/E3** — either start from
`deepsight_randinit` (real headroom, but won't converge to SOTA in budget — the *λ-delta* is
still valid since both arms start identically), **or** start from the released ckpt but
**evaluate on the weakest scenarios** (today's run shows `YieldToEmergencyVehicle_*` at
0.12–0.19 vs straight scenes at 0.02 — there's headroom there). Record which start point was
used; it changes interpretation.

---

### E1 — Tiny-set overfit (plumbing + capacity)

- **Question / hypothesis.** Can the optimizer drive **both** `loss_rec` and `loss_gen`
  toward ~0 on a handful of fixed samples? If yes → gradients flow end-to-end through the
  fused sequence, the `vis_head`→DINOv3 MSE path is differentiable, and the collator's
  `label_bev_masks`/`bevs_masks`/`template_mask` select the right positions. If `loss_gen`
  *can't* be overfit, the world head is mis-wired (a bug JEPA would inherit).
- **Design.** N = 8–64 fixed samples, **1 GPU, no DeepSpeed**, batch 1 (+ small grad-accum),
  **constant LR** (try 1e-4), `max_steps` ≈ 200–500, overfit the same batch repeatedly. Base
  off [ad_bev_train_smoke.yaml](configs/ad_bev_train_smoke.yaml) → new `configs/ad_bev_overfit.yaml`
  (`max_steps` up, scheduler `constant`, saving off, `overwrite_cache`).
- **Start point (refined).** Default **`deepsight_randinit`** — the canonical "overfit one
  batch" test is most decisive from random weights (`loss_rec` ≈12.5→~0, `loss_gen`
  high→low proves the whole gradient path + capacity). From the released ckpt the loss
  already starts near-zero on this in-distribution data, so it's a weaker signal; that
  variant (one-line `model_name_or_path` swap) instead tests fine-tune descent.
- **Measure.** `loss`, `loss_rec`, `loss_gen` curves. The trainer logs only total `loss`
  (→ `training_loss.png` via `plot_loss`); the patched model **prints** `loss/loss_rec/loss_gen`
  per step, so the run is tee'd to `saves/ad_bev_overfit/run.log` and the world-loss curve is
  grepped from there.
- **Success / interpretation.** Both curves decrease monotonically to small values;
  `loss_gen` drops well below its step-0 value (and below the E3 predict-mean floor). PASS →
  pipeline is sound, proceed to E2. FAIL (loss_gen flat / NaN / not decreasing) → debug the
  world-head wiring or masks *before* anything else.
- **Cost.** Minutes on 1 GPU (~300 steps × ~1–2 s/step).
- **Files — BUILT (2026-06-13), zero original files touched.** `configs/ad_bev_overfit.yaml`;
  a private dataset_dir `local_data/e1_overfit/` holding `overfit_samples.jsonl` (16 fixed
  samples = `head -16 train_samples.jsonl`, all from `AccidentTwoWays_…Route1102`) + its own
  `dataset_info.json`. No model-code change. **Not yet run** — launch:
  `CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/ad_bev_overfit.yaml 2>&1 | tee saves/ad_bev_overfit/run.log`.

### E2 — World-loss ablation (does the world objective help the policy?)

Split into **two sub-experiments** (user decision 2026-06-15), each with λ∈{0,2} arms (4 runs
total), short ≤6h combined:
- **E2-1 (random-init):** both arms from `deepsight_randinit` (identical start) → the held-out
  L2 *delta* is a causally clean attribution to `loss_gen`. Absolute L2 will be poor
  (undertrained 3B) and randinit may emit some unparseable trajectories — only the delta counts.
- **E2-2 (released ckpt):** both arms continue-finetune from `checkpoints/deepsight` → realistic
  trajectories/L2. Caveat: that ckpt is already world-loss-shaped, so λ=0 continued-FT doesn't
  undo it — a weaker/contaminated test; evaluate on the weakest scenarios for sensitivity.

- **Question / hypothesis.** Does the world loss improve the **action output**? Two short
  runs identical except the world weight: **`λ_world = 0`** (text-only) vs **`λ_world = 2`**
  (stock). Compare **held-out open-loop L2**.
- **Design.** Same train subset, same seed / steps / LR / batch for both arms; only
  `world_loss_weight` differs (set in the training YAML — knob now implemented, see above).
  λ=0 config built: [configs/ad_bev_overfit_lambda0.yaml](configs/ad_bev_overfit_lambda0.yaml)
  (overfit template; the real E2 arms will be train-data configs). After each run, infer on the
  held-out split and score with `eval_l2.py`. Keep runs short (relative comparison, not a repro).
- **Start point.** E2-1 randinit / E2-2 released (both arms share the identical start within each).
- **Measure.** Held-out 1s/2s L2 for λ=0 vs λ=2; secondary: `loss_rec` trajectory-token CE,
  and the loss curves.
- **Success / interpretation.**
  - `λ=2` meaningfully **better** than `λ=0` → the world objective genuinely shapes the
    policy → **JEPA = upgrade to a working component** (strong green light).
  - **Tie** (within noise) → the original world model is **decorative** — consistent with
    `merge_model_weight.py` *stripping* `vis_head`/`dino` for serving and with the world head
    being unread at inference. Then JEPA must justify its value differently (e.g. via the
    action-conditioned / dynamics-aware angle), or the head is a candidate to drop.
- **Cost.** Two short subset runs (multi-GPU optional).
- **Files to add (later).** The `λ_world` env knob (patched modeling copy via the shim);
  `configs/ad_bev_ablate_lambda0.yaml`; `local_data/heldout_scenes.txt` + its infer JSONL.

### E3 — World-loss learning curve vs a trivial floor

- **Question / hypothesis.** Does `loss_gen` actually predict **scene-specific** futures, or
  does it **collapse toward the mean** DINOv3 feature? (The frozen-MSE target can be
  dominated by a few high-variance channels / admit partial collapse — see
  [WORLD_MODEL_JEPA.md §4.1](WORLD_MODEL_JEPA.md).)
- **Design.** On a modest subset, compare trained `loss_gen` to cheap baselines computed
  offline from the DINOv3 targets: **(a) predict-the-mean** floor = MSE of every prediction
  vs the dataset-mean DINOv3 feature; **(b) random `vis_head`** MSE (step-0). Optionally add
  a **collapse check**: variance of `vis_embeds` across samples (VICReg-style) and per-token
  cosine spread — low variance ⇒ collapse. E3 can **piggyback on E2's `λ=2` run logs** (no
  separate training), plus one small offline script for the floor.
- **Measure.** Ratio `trained_loss_gen / predict_mean_floor`; prediction variance; per-token
  cosine to target.
- **Success / interpretation.** Trained `loss_gen` **well below** the predict-mean floor +
  healthy prediction variance ⇒ the target is genuinely learned. Near the floor / low
  variance ⇒ partial collapse — which *strengthens* the JEPA case (its anti-collapse
  machinery + in-domain target are precisely the fix) and informs the **features-vs-occupancy**
  target choice.
- **Cost.** Cheap — mostly an offline floor computation reusing the collator's frozen-DINOv3
  path; reads E2's training logs.
- **Files to add (later).** A small `src/tools/dino_target_floor.py` (compute mean/variance
  floors over a subset, reusing `ad_collator`'s DINOv3 target prep); a log-parse for the curve.

**Recommended order.** **E1 → E2 (→ E3 piggybacked).** E1 (minutes) guards against a wiring
bug; E2 answers the decisive pre-JEPA question; E3 sharpens E2's interpretation. JEPA work
starts only after these read out.

---

## Open Questions / Next Steps

- [x] **Inference at scale = integration check.** Ran the multi-GPU inferer over the
      stride-25 base subset (8779 samples); merged open-loop L2 = **0.148** (1s 0.108 / 2s
      0.188). Integration **passed**; the number is ~4× below the paper's 0.58 — optimistic,
      attributed mainly to probable base/train overlap + a not-yet-paper-matched L2
      convention (see 2026-06-12 log).
- [ ] **Tighten the eval before quoting a reproduction number.** Use a genuinely held-out
      split (or verify base ∉ training) and match `eval_l2.py`'s horizon/averaging to the
      paper's so the L2 is comparable. This becomes the baseline JEPA must not regress.
- [ ] **Short real training run** (few hundred steps, 4-GPU ZeRO-2): confirm `loss_gen`
      *decreases* (world head is genuinely supervised), grad_norm finite, and a resulting
      checkpoint re-runs through inference.
- [ ] **Instrument the two JEPA seams** before editing: capture baseline `loss_gen`
      magnitude (and optionally world-head action sensitivity) so the JEPA swap is a
      localized, measurable change.
- [ ] **JEPA implementation:** replace the frozen DINOv3 target with an EMA in-domain BEV
      encoder + stop-grad + variance/predictor anti-collapse; later replace the top-down
      source with an onboard surround-cam BEV (or occupancy/flow) target. Open design
      choice: **features vs occupancy** as the target.

---

## Daily Log

### 2026-06-12 — end-to-end integration check via an open-loop validation run

**Purpose of today's work:** verify that the **assembled repo works as a whole** — i.e.
that all the files we *added* and *updated* to fill the original project's missing parts
(the transformers shim + dinov3-config fix in `setup_local_inference.py`, the local data
builders, `infer_local.py` / `infer_local_multi_gpu.py`, the `*_local` eval copies, the
training stubs + configs, the one-line registry repoint) cooperate correctly when run
together against the released checkpoint. Piece-by-piece smoke tests on earlier days had
already shown the parts work in isolation (single-scene inference L2 ≈ 0.043/0.066 m, the
1-step training smoke, the 4-GPU ZeRO-2 step). What was still unconfirmed was **holistic
validity**: do they hold up across a real, diverse evaluation?

**The check itself (the design / intent):** run the full inference→merge→eval pipeline on
the **Bench2Drive validation data** and compare the open-loop L2 to the paper's **0.58**.
Rather than the entire original validation set (consecutive 10 Hz frames are near-duplicate
and would burn compute for little extra signal), I built a **stride-25 subset** of it —
this keeps scenario diversity while cutting frame redundancy ~25×. The expectation is a
result **somehow close to the paper's 0.58** (likely a touch lower, since `bench2drive_base`
probably overlaps the checkpoint's training data); landing in that range is what
"the integration is valid" means here. If it does, every added/updated file is confirmed
to interoperate end-to-end and the repo is trustworthy enough to start the JEPA redesign on.

**Supporting work that made the check runnable:**
- Closed out the **multi-GPU inference launcher** (`src/infer_local_multi_gpu.py`), the tool
  that actually executes the validation run: added `--models-per-gpu` (GPU packing — an
  ~8 GB model fits several times on an 80 GB A100 and a bs=1 worker rarely saturates the
  GPU) and `--stagger` (spreads the simultaneous weight-load disk/RAM spike; startup-only,
  no effect on results). Verified the eval gate: the blocking `p.wait()` loop waits for
  **all** shards (total time = the slowest worker) and a failed shard aborts the merge — so
  `eval_l2.py` runs exactly once, on the complete merged output.
- Documented both inference forms in `RUN_LOCAL_INFERENCE.md` (single manual commands kept;
  the one-shot automatic command added) with a "reproduce open-loop ≈0.58" recipe that
  encodes today's intent: favor scenario diversity over frame density via a coarse stride.

**Adjacent confirmations (not the main check):**
- The **random-init** path (`scripts/make_random_init.py` → `checkpoints/deepsight_randinit`)
  reuses the released config/tokenizer but constructs the model from config (no
  `from_pretrained`); `loss_rec ≈ ln(vocab)` confirms truly random weights. Noted it is
  **not** paper-comparable (a 3B VLM from random init needs web-scale pretraining).
- Consolidated the **conceptual background** into this log from `CLAUDE.md` /
  `SRC_CODE_MAP.md` and `WORLD_MODEL_JEPA.md`, and pinned the exact code seams the JEPA swap
  will touch — so once the integration is confirmed valid, the redesign is a localized edit.
- Re-confirmed the **file-edit discipline**: all changes in `*_local` copies / new files;
  the only upstream edit is the one-line `data/dataset_info.json` repoint.

**Result of the validation run (ALL SCENES).** The stride-25 subset produced **8780**
inference samples (**8779 parsed OK**, 1 unparseable/short); merged open-loop L2:

```
Period   Samples   Mean L2     Std Dev    Min        Max
------------------------------------------------------------
1s       8779      0.108303    0.193676   0.000000   3.693325
2s       8779      0.188458    0.355573   0.000000   7.022870
avg (overall): 0.148380
```

Per-scene the spread is wide — easy near-straight scenes sit at ~0.02–0.06 m
(e.g. `AccidentTwoWays_…Route1103` avg 0.022) while interactive/long-tail ones climb to
~0.12–0.16 m (e.g. the `YieldToEmergencyVehicle_…` family, 0.08–0.16). The heavy tails
(max 3.69 m @1s, 7.02 m @2s; std > mean) confirm a minority of hard frames dominate the
upper range while the bulk is easy.

**Integration verdict: PASS.** All added/updated files cooperate end-to-end across 8.8k
diverse samples — the pipeline parsed 8779/8780 with no crashes, so the assembled repo is
confirmed working as a unit. The numeric goal ("somehow close to 0.58") is *not* matched in
the expected direction, though: **0.148 vs 0.58 — mine is ~4× lower (better)**.

**Why mine ≠ the paper's 0.58 (justification).** A lower L2 than the original is the
opposite of a bug-induced regression; it almost always means the eval is *easier* than the
paper's, for several compounding reasons (in rough order of impact):

1. **Train/eval overlap (data leakage) — the dominant factor.** The released checkpoint was
   trained on Bench2Drive, and my subset is drawn from `bench2drive_base`, which very likely
   overlaps that training data. The model is being scored on frames it effectively saw, so
   it near-memorizes the expert future → unrealistically low L2. The paper's 0.58 is on a
   *held-out* split. This alone can explain a multiple-× gap.
2. **Horizon / averaging-convention mismatch.** My `eval_l2.py` reports only **1s and 2s**
   (DeepSight predicts 4 waypoints = 2 s) and averages those two. If the paper's 0.58 folds
   in a longer/denser horizon or a different per-waypoint vs per-endpoint averaging
   convention (the L2 grows fast with horizon — note my 2s is already ~1.7× my 1s), the two
   numbers are not the same metric. I have not byte-matched my averaging to the paper's.
3. **Subset composition skew.** Even at stride 25 the base scenes skew toward
   low-curvature, near-constant-velocity driving where the next 2 s are almost
   deterministic (many per-sample mins are 0.000). My diversity-over-density sampling
   improves coverage but the frame *mix* still differs from the paper's evaluation set, and
   easy frames pull the mean down.
4. **GT/coordinate provenance.** My ground-truth waypoints come from the same Bench2Drive
   logs used to build the prompt (history + target-pixel goal), so on straight segments the
   answer is strongly constrained by the inputs — a partly self-consistent, "easy" target.

**Takeaway:** the run validates **integration** (everything runs together and produces
sane, parseable, scenario-sensible trajectories) but **not** a clean paper-reproduction —
the 0.148 is optimistic mainly due to probable train/base overlap and a not-yet-aligned L2
convention. To turn this into a real reproduction I'd need a genuinely held-out split (or
confirmation that base ∉ training) and an averaging convention matched to the paper.

**Next:** with integration confirmed, the JEPA changes (see *Open Questions*) can begin;
separately, tighten the eval (held-out split + paper-matched L2 averaging) before quoting
any reproduction number.

### 2026-06-13 — designed trainability experiments (E1/E2/E3) before JEPA

**TODO — to implement on request (designed today, nothing built yet):**
- [x] **E1 — tiny-set overfit** (plumbing + capacity: can `loss_rec` *and* `loss_gen` be driven to ~0?) — **PASS**, see results at end of this entry.
- [ ] **E2 — world-loss ablation** (`λ_world ∈ {0, 2}`: does the world objective actually improve held-out trajectory L2?)
- [ ] **E3 — world-loss learning curve vs trivial floor** (does `loss_gen` learn structure or collapse to the mean?)

**Purpose.** Yesterday's integration check confirmed the assembled repo *runs* end-to-end,
but it does not tell us whether the **original DINOv3-supervised recipe is effectively
trainable** — the thing we must know before deciding JEPA is an *upgrade* vs a *replacement*.
Training on Bench2Drive-full to find out is far too slow, so today I **designed three small,
controlled experiments** that each answer one facet on a tiny slice of data, with explicit
success criteria and JEPA implications. Full, implementation-ready specs (shared infra, the
`λ_world` env knob, start-checkpoint caveat, per-experiment design / measurements / cost /
files) are written up in
[§ Planned Experiments — is the original recipe effectively trainable?](#planned-experiments--is-the-original-recipe-effectively-trainable-e1e2e3).

- **E1** isolates plumbing/capacity (gradients flow, `vis_head`→DINOv3 path differentiable,
  collator masks correct) — a bug here would also break JEPA, which reuses the same path.
- **E2** is the decisive one: λ=2 beating λ=0 on held-out L2 ⇒ JEPA upgrades a *working*
  component; a tie ⇒ the world head is *decorative* (matches it being stripped for serving and
  unread at inference), reframing JEPA's value proposition.
- **E3** checks whether the frozen-DINOv3 target is genuinely learned or collapses toward the
  mean — directly informing the JEPA anti-collapse design and the features-vs-occupancy choice.

**Recommended order:** E1 → E2 (→ E3 piggybacked on E2's λ=2 logs). **Nothing implemented;**
awaiting the go-ahead to build the shared infra + per-experiment configs/scripts.

---

#### E1 RESULT — PASS

Setup built with zero edits to original files: `configs/ad_bev_overfit.yaml` + a private
`local_data/e1_overfit/` (16 fixed samples = `head -16 train_samples.jsonl`, all from
`AccidentTwoWays_…Route1102`, + its own `dataset_info.json`). Start = `deepsight_randinit`,
full finetune, constant lr 1e-4, **300 steps**, batch 1 (1 sample/step). 300 steps over 16
samples = **18.75 epochs** (300/16) — i.e. the fixed set was seen ~19 times; "step" here =
one optimizer update on one sample, not a pass over the data.

Total-loss curve (from `saves/ad_bev_overfit/.../trainer_log.jsonl`; ~23 min on 1 GPU):

```
step    1   ~16.08      (mean steps 1–5 = 11.88)
step   50    0.79
step  150    0.07
step  300    0.034       (min over run = 0.021;  last-20 mean = 0.227, noisy)
```

**Both losses reached ~0 — provable without the per-step split.** The trainer logs only the
total `loss`; the `loss_rec`/`loss_gen` breakdown is printed to stdout (not captured this
run — fixed going forward by `scripts/train.sh` auto-logging). But since
`loss = loss_rec + 2·loss_gen` with **both terms ≥ 0**:
- final total 0.034 ⇒ `loss_rec ≤ 0.034` **and** `loss_gen ≤ 0.017`;
- start: total 16.08 with random-init `loss_rec ≈ ln(vocab) ≈ 12.5` ⇒ initial `loss_gen ≈ 1.8`.

So `loss_gen` fell ≈1.8 → ≤0.017 (~100×) and `loss_rec` ≈12.5 → ≤0.034. **Conclusion:**
gradients flow end-to-end through the fused sequence, the `vis_head`→DINOv3-MSE branch is
differentiable and learnable, and the AD-collator masks select the right `<|bev_token_i|>`
positions — **no plumbing bug; capacity sufficient.** The noisy descent is expected
(batch 1, single-sample steps cycling 16 samples), not instability; the smoothed
`training_loss.png` is monotone. Caveat: this proves *capacity*, not generalization (that's
E2/E3). **Pipeline sound → proceed to E2.**

> Tooling added alongside this result: `scripts/train.sh` (launcher) + per-config private
> `dataset_dir`s. Its save behavior was reworked on 2026-06-15 — see that day's log entry.

### 2026-06-15 — training-launch tooling: config-driven saving, 3-loss plot, general multi-GPU

Hardened the E-experiment training harness (no original repo files touched):

- **Saving is now config-driven, not wrapper-driven.** The earlier approach had
  `llamafactory-cli` write the model and the wrapper delete it via a magic `# SAVE_MODEL`
  comment — unintuitive (a commented line silently acting) and wasteful (write-then-delete).
  Removed it. Saving is controlled purely by real config args: all three configs use
  `save_strategy: "no"` → the workflow's single unconditional `trainer.save_model()`
  ([workflow.py:100](src/llamafactory/train/sft/workflow.py#L100)) writes **exactly one**
  final model, no `checkpoint-*/` dirs. Note: there is **no** config that saves *zero*
  models (that final save is unconditional) — throwaway runs are `rm -rf`'d manually.
  `ad_bev_train_local.yaml` now carries a **commented** block of alternative strategies
  (rolling checkpoints / per-epoch / best-by-eval-loss) to uncomment when needed.
- **`scripts/train.sh` is now a thin wrapper:** timestamped run dir
  `saves/<exp>/<unixtime>_<exp>/`, always-on `run.log`, and the 3-loss plot. It no longer
  parses directives or touches weights.
- **3-loss plot** (`scripts/plot_losses.py`): parses the model's per-forward
  `loss/loss_rec/loss_gen` prints from `run.log` → `losses_split.png` (three descending
  curves, log-y) — recovering the rec/gen split the trainer doesn't log, without editing the
  model/trainer.
- **General multi-GPU** documented in [RUN_LOCAL_TRAINING.md](RUN_LOCAL_TRAINING.md) (Step 3):
  any config on N GPUs via `CUDA_VISIBLE_DEVICES=<list> FORCE_TORCHRUN=1 scripts/train.sh
  <config>` (CLI auto-launches torchrun→DDP; DeepSpeed optional). So E1 on 2 GPUs is just the
  general pattern applied to `configs/ad_bev_overfit.yaml`.

Net: configs are lean (usage-only; rationale lives in the RUN/RESEARCH docs), saving is
predictable and standard, and every run self-documents via `run.log` + `losses_split.png`.

### E1 re-verified (PASS); λ_world knob implemented (config-driven); E2 scoped

- **E1 full re-check: PASS.** Audited every E1 file (config, registry, the 30-sample jsonl —
  all 15-img paths exist, 6 scenes × 5; train.sh + plot_losses syntax OK; modeling print
  intact). Run `1781518102_…`: `loss_rec` 12.3→min 0.0014, `loss_gen` 1.78→min 0.017, total
  min 0.078 — both losses driven down on the *diverse* set. Pipeline sound.
- **λ_world knob — config-driven (not env/shim).** After weighing options the user chose a
  minimal 1-line edit over a 1500-line file copy. Implemented as a real YAML arg
  `world_loss_weight` (default 2.0 = unchanged behavior) via 3 comment-and-add edits:
  `FinetuningArguments` field, wiring in `train/sft/workflow.py`
  (`model.config.world_loss_weight = finetuning_args.world_loss_weight`), and the read in
  `modeling_qwen2_5_vl.py` (`getattr(self.config, "world_loss_weight", 2.0)`; original line
  kept commented). **Verified end-to-end without training:** parser accepts the YAML key and
  it flows to `finetuning_args` (0.0 for the λ=0 config, 2.0 for λ=2); all originals still
  compile. Set λ=2 explicitly in `ad_bev_{overfit,train_smoke,train_local}.yaml`; added the
  λ=0 ablation config `configs/ad_bev_overfit_lambda0.yaml`. (Earlier env-var/shim version was
  reverted.)
- **E2 scoped** into **E2-1 (random-init, clean delta)** and **E2-2 (released ckpt, realistic)**,
  λ∈{0,2} each, short ≤6h budget — see the Planned-Experiments E2 spec above.
- **Next:** build the disjoint **train / held-out** split + the E2 train-data configs (the
  current λ=0 config is the overfit template), then run the 4 arms and compare held-out L2.
  Optional final knob confirmation: a 1-step `scripts/train.sh configs/ad_bev_overfit_lambda0.yaml`
  should print `loss == loss_rec`.

### E2 finalized — 4 configs, 2000/500 data split, auto-eval wired (design; no results yet)

The E2 ablation is now fully set up and ready to run (results to be discussed in a later log).

**The 4 runs** = 2 sub-experiments × 2 λ arms, identical within a pair except `world_loss_weight`:

| Config | Init | `world_loss_weight` |
|---|---|---|
| `configs/ad_bev_overfit_lambda2_randinit.yaml` | `deepsight_randinit` | 2.0 |
| `configs/ad_bev_overfit_lambda0_randinit.yaml` | `deepsight_randinit` | 0.0 |
| `configs/ad_bev_overfit_lambda2_preinit.yaml`  | `checkpoints/deepsight` (released) | 2.0 |
| `configs/ad_bev_overfit_lambda0_preinit.yaml`  | `checkpoints/deepsight` (released) | 0.0 |

- **E2-1** = the two `*_randinit` arms (clean λ-delta from identical random init; absolute L2
  will be poor, only the delta is meaningful). **E2-2** = the two `*_preinit` arms (realistic
  L2 from the released model; weaker test since that ckpt is already world-loss-shaped).
- Shared schedule: full finetune, `num_train_epochs: 2`, lr `1e-4` constant + 10 warmup steps,
  bs 1, `save_strategy: "no"` (one final model). lr is uniform across all 4 (ablation valid);
  for the preinit pair `1e-4` is aggressive — drop both to `2e-5` if they degrade vs base.

**Data** (`local_data/e2_overfit_lambda/`, registry `dataset_info.json` → the train file):
- `overfit_samples_bigger.jsonl` — **2000** train samples (15-img), 110 scenes, 39 scenario types.
- `heldout_infer.jsonl` — **500** held-out samples (10-img), 45 scenes, 24 scenario types.
- Built from a **disjoint scene split** (train = shuffled `ready_scenes`[:110], held-out =
  [110:155]) → verified **0 sample overlap**. More data + a genuinely unseen, diverse held-out
  set = lower-variance, generalization-measuring L2 (the right signal for the λ ablation).

**Auto-eval** (`scripts/train.sh --eval <heldout.jsonl>`): after training, the wrapper runs
inference with the just-saved checkpoint on the held-out set + `eval_l2.py`, writing results
**into the run dir** (`saves/<arm>/<unixtime>_<arm>/`: `heldout_infer.json`, `eval_plots/`,
`eval.log`) — not `debug/`. (`src/infer_local_multi_gpu.py` auto-shards across the visible GPUs.)

**Launch (per arm; multi-GPU needs the ZeRO override):**
```
CUDA_VISIBLE_DEVICES=0,1,2 FORCE_TORCHRUN=1 scripts/train.sh \
    configs/ad_bev_overfit_lambda2_randinit.yaml \
    --eval local_data/e2_overfit_lambda/heldout_infer.jsonl \
    deepspeed=examples/deepspeed/ds_z2_config.json
```
**Estimated cost (3 GPUs, ZeRO-2):** ~1.5–1.7 h training + ~14 min eval per arm → ~7–8 h for
all 4 (≈4 h if `num_train_epochs: 1`, or run E2-1 and E2-2 as two separate sessions).

**Read-out plan:** compare held-out 1s/2s L2 of λ=2 vs λ=0 *within each pair*. λ=2 better ⇒
world loss helps the policy (JEPA upgrades a working component); tie ⇒ decorative (reframes JEPA).

### E2 RESULTS (4 arms): world loss helps in the competent regime, inconclusive from scratch

All 4 arms ran with auto-eval. Train losses from `run.log`; held-out L2 (500 unseen samples)
from `eval.log` (`ALL SCENES`).

| Sub-exp / arm | train `loss_rec` (last-50) | train `loss_gen` (last-50) | held-out L2 1s | 2s | **avg** | parsed |
|---|---|---|---|---|---|---|
| **E2-1** randinit **λ=0** | 0.517 | **1.82** (untrained) | 1.034 | 3.032 | **2.033** | 339/500 |
| **E2-1** randinit **λ=2** | 0.530 | 0.27 (trained ↓) | 1.459 | 3.039 | **2.249** | 307/500 |
| **E2-2** preinit **λ=0** | 0.327 | **0.49** (drifted ↑) | 0.790 | 1.808 | **1.299** | 500/500 |
| **E2-2** preinit **λ=2** | 0.303 | 0.04 (kept low) | 0.700 | 1.583 | **1.142** | 500/500 |

**Knob sanity — passes.** λ=0 → `loss_gen` gets no gradient (randinit leaves it ~1.8; preinit
lets it **drift up** 0.016→0.49 as trajectory-only FT pulls representations off the DINOv3
targets). λ=2 → `loss_gen` optimized down. `loss_rec` reaches a similar low in both arms of a
pair → trajectory-fit capacity matched; held-out differences are about **generalization**.

**E2-1 (random-init): inconclusive (as predicted).** Both arms near-useless on held-out
(avg ~2.0–2.2 m; 2s ≈ 3 m), and 161/193 of 500 samples **unparseable** — so the L2s are
averaged over *different* subsets (339 vs 307) and aren't comparable. 2000 samples / 2 epochs
is far too little for a 3B model from scratch → treat E2-1 as **null**.

**E2-2 (released ckpt): λ=2 measurably better — clean signal.** Both arms parsed **500/500**
(identical set → comparable). World loss lowers held-out L2 across both horizons:
1s 0.700 vs 0.790, 2s 1.583 vs 1.808, **avg 1.142 vs 1.299 → ≈12% lower with λ=2**. Mechanism
is visible in training: with λ=0 the inherited world representation **degrades** (`loss_gen`
0.016→0.49) and L2 worsens; with λ=2 it's preserved (0.04) and L2 improves. So the world loss
acts as a **representation regularizer that keeps the policy dynamics-aware** during fine-tuning.

**Expectations vs. outcome.** E2-1 matched the prediction (delta-only, unparseable, poor
absolute → uninformative). E2-2: predicted *"realistic L2 but weaker/contaminated, effect
likely small"*; the effect was **clearer than expected (~12%, both horizons, clean 500/500)**.
The contamination (released model already world-loss-trained) didn't wash the signal out — it
surfaced as *degradation-on-removal* (λ=0 lets the world representation rot, which held-out L2
catches).

**Verdict for JEPA. ⚠️ SUPERSEDED — see "E2 re-analysis" below.** (Originally read: "world
objective is not decorative → green light for JEPA." On reflection this over-claimed: the E2-2
comparison is confounded by a world-loss-pretrained init, so it does **not** establish that the
world objective helps the policy. Retracted; corrected design below.)

**Caveats / follow-ups.** (1) No untrained-base reference — both preinit arms FT'd at lr 1e-4
and may have *degraded* vs base `checkpoints/deepsight`; eval the base on this held-out set to
know whether the λ gain is "less degradation" vs "real gain". (2) Lower preinit lr to 2e-5 to
cut forgetting and sharpen the signal. (3) E2-1 needs much more compute (or LoRA-on-frozen) for
a clean from-scratch answer.

### 2026-06-16 — E2 re-analysis: the result does NOT prove the claim; a properly controlled design

**Why E2-2 does not establish "the world objective helps the policy."** The released
checkpoint was **already trained with the world loss**, so its useful, dynamics-aware
representations are *pre-baked*. The two arms therefore compare:
- **λ=2:** keep the world loss → those pre-baked representations are **preserved**;
- **λ=0:** drop it → they **drift/rot** (training shows `loss_gen` 0.016 → 0.49).

So the ≈12% held-out gap measures **"how much removing the world loss damages an
already-world-trained model,"** not **"how much the world loss adds."** Two further holes:
there is **no absolute baseline** (the un-fine-tuned base was never scored on the held-out set,
so we can't tell whether λ=2 *improved over base* or merely *degraded less*), and it is a
**single seed** (a 12% gap can be seed noise). E2-1, which *had* a neutral (random) init, was
too undertrained to learn the task (high unparseable, null delta). **Net: neither arm answers
the question.** The earlier "green light" verdict is retracted.

**The core flaw is the starting point.** A valid ablation changes only the variable under test
*and* starts from an init that is **neutral with respect to that variable**. A model already
trained with the world loss is not a neutral baseline for testing the world loss. Everything
else in E2 (same data/schedule, only λ toggled, same pipeline) was correct — only the init was
contaminated.

**Required properties of a valid, *comparable* ablation:** (a) init **neutral** w.r.t. the
world loss; (b) init **capable** enough to learn the task within budget (else null, like E2-1);
(c) identical architecture across arms — both keep the BEV tokens + `vis_head`, toggling only
the `loss_gen` *supervision*; (d) identical data/compute, **≥2–3 seeds** (auxiliary-loss effects
are small/noisy); (e) always report the **base (no-train) reference** so direction is visible.

**Design options considered (and why the chosen one wins):**

| Option | Neutral init? | Capable in budget? | Verdict |
|---|---|---|---|
| Full **randinit** from scratch, more compute | ✅ | ❌ (3B from scratch needs web-scale; E2-1 already null) | infeasible |
| **Frozen backbone + LoRA** from released ckpt | ❌ (frozen features already world-shaped) | ✅ | confounded — same flaw as E2-2 |
| **"Wash out" world loss** from released, then branch | ⚠️ ill-defined ("how washed?") | ✅ | arbitrary; rejected |
| **Warm-start from base Qwen2.5-VL-3B** + random DeepSight heads | ✅ (base VLM never saw the world loss/BEV task) | ✅ (pretrained → converges fast on small data) | **chosen** |

**Chosen approach — E2′ (neutral-capable init).** Build the init the way the authors did
*before* their training: take **base Qwen2.5-VL-3B** (pretrained general VLM — capable but
task-neutral), graft it into the DeepSight architecture (LLM + vision tower from base; resize
embeddings for the added `<|bev_token|>`/`<|pixel_token|>` rows = random; `vis_head` random;
**frozen pretrained DINOv3**). Then run the **identical** protocol, toggling only λ∈{0,2}:
- same train/held-out split, same compute, **multiple seeds**;
- both arms have the BEV tokens + `vis_head` in-graph — only `loss_gen` supervision differs;
- report `base` (no-train) vs `λ=0` vs `λ=2` held-out L2.

Because the init is neutral, any λ effect is attributable to the world objective; because the
backbone is *pretrained*, both arms can actually learn the task in our budget (unlike randinit).
This also becomes the **fixed protocol** for all later changes (JEPA target, action-conditioning,
…): same init/data/compute/seeds/eval, vary one component → every result is apples-to-apples.

**Residual honesty.** Even E2′ tests "does the world loss help when fine-tuning a pretrained
VLM on a *small* driving set" — not "at DeepSight's full training scale" (only the authors'
scale could show that). The effect may also need **more train data** than 2000 to surface
(auxiliary-loss benefits often grow with data). Both are acceptable, stated limitations.

**Status: design only — to implement on request.** Build steps will be: (1) download base
Qwen2.5-VL-3B; (2) a `make_warmstart_init.py` (load base weights into the DeepSight arch +
resize embeddings + random heads + frozen pretrained DINOv3); (3) E2′ configs (λ0/λ2, ≥2 seeds);
(4) base-reference eval; (5) run via the fixed protocol + auto-eval.

### E2-3: concrete build + run plan (warm-start neutral-capable init)

E2-3 is the executable form of the E2′ design above. Goal restated in one line: **with a
*neutral-but-capable* init, does adding the DINOv3 world loss (λ=2) beat not adding it (λ=0),
both measured against the no-train base?** Below is grounded in the actual checkpoints (verified
2026-06-16), so the grafting is exact rather than hand-wavy.

**Architecture inventory (from `checkpoints/deepsight/`).** Weight groups in the released
checkpoint: `model.*` (434 keys = Qwen2.5 LLM), `visual.*` (390 = Qwen vision tower),
`dinov3.*` (415 = frozen DINOv3 target extractor), `lm_head.weight`, `vis_head.weight`
(the 2048→1024 world-latent head). Config: `vocab_size = 153536`, `hidden_size = 2048`,
`tie_word_embeddings = None` (⇒ `lm_head` is **untied** — must be grafted separately from
`embed_tokens`). Base Qwen2.5-VL-3B vocab = 151936 ⇒ **resize delta = 1600 rows.**

**⚠ The vocab is NOT a clean append** (verified): of 1305 `<|bev_token|>` rows, **265 reuse
base's reserved tail** (ids 151671–151935, inside the base 151936 range — these are Qwen's
unused/reserved padding slots) and the rest, plus all `<|pixel_token|>` (511) and `<CoT_flag_*>`
etc., occupy the **1600 genuinely-new** rows (ids 151936–153535). Implication for the graft:
copy base rows `[0:151936]` wholesale into the DeepSight `embed_tokens`/`lm_head` and random-init
only rows `[151936:153536]`. The 265 bev tokens sitting in `[0:151936]` thus inherit base's
*reserved-row* embeddings — harmless (bev tokens are learnable placeholders) and **identical
across both arms**, so it cannot bias the ablation.

**Init recipe — `scripts/make_warmstart_init.py`** (new file; modeled on `make_random_init.py`,
which already proves the config-construct + selective-load pattern). Build the DeepSight arch
from the released `config.json` (gets vocab 153536, `dinov3_config`, `visual_target_dim`, etc.),
then populate weights per-group:

| Weight group | Source in E2-3 | Rationale |
|---|---|---|
| `model.*` (LLM) | **base Qwen2.5-VL-3B** | capable, world-loss-neutral |
| `visual.*` (vision tower) | **base Qwen2.5-VL-3B** | same |
| `embed_tokens` / `lm_head` | base rows `[0:151936]`; rows `[151936:153536]` **random** | new bev/pixel/CoT tokens unseen by base |
| `vis_head` | **random** | world head must be neutral (never pretrained) |
| `dinov3.*` | **pretrained**, loaded from `checkpoints/deepsight/` (`--keep-dino-pretrained` logic) | it is *Meta's frozen DINOv3 target extractor*, never trained by DeepSight ⇒ neutral w.r.t. the world **loss**; needed so `loss_gen` has a meaningful (not random) target |

Tokenizer/processor copied from `checkpoints/deepsight/` (so the bev/pixel vocab is already
correct — same trick `make_random_init.py` uses to avoid hand-registering tokens). Save with
`--seed` so the random rows (`vis_head`, new-token embeddings) are reproducible per seed.

> Why pretrained DINOv3 is still "neutral": the contamination in E2-2 was that the **LLM/heads**
> had already been *trained by the world loss*. DINOv3 here is only the fixed feature *target*;
> using Meta's pretrained weights is exactly what a from-the-authors'-start init would do, and it
> is identical across λ=0 and λ=2 arms. (λ=0 simply never consults it.)

**Arms & seeds (5 evaluations).**

| Arm | Init | λ_world | Train? |
|---|---|---|---|
| `base` (reference) | warm-start init, **no training** | — | no |
| `λ0/seedA`, `λ0/seedB` | warm-start init | 0.0 | yes |
| `λ2/seedA`, `λ2/seedB` | warm-start init | 2.0 | yes |

Two seeds per λ (seeds {0,1}) → 4 training runs + 1 no-train eval. Seeds vary **both** the random
graft rows *and* the data-shuffle/trainer seed. (Stretch: add seed 2 → 6 runs if time allows;
auxiliary-loss effects are small, so ≥2 seeds is the floor for believability.)

**Configs.** Clone the E2 configs to `configs/ad_bev_e2_3_lambda{0,2}_seed{0,1}.yaml`. Identical
across all arms **except** `world_loss_weight` and `seed`:
`model_name_or_path: checkpoints/deepsight_warmstart`; `finetuning_type: full`;
`freeze_vision_tower: false` (matches the released recipe **and** E2-2, so results transfer —
the vision tower is the same pretrained one in every arm, so this stays a controlled variable);
dataset = the same 2000-train / 445-held-out `e2_overfit_lambda` registry; `num_train_epochs: 2`;
`lr: 1.0e-4`; `lr_scheduler: constant`; `warmup_steps: 10`; `save_strategy: "no"` (one final
model); `seed: <0|1>`. DINOv3 stays frozen as in the released recipe (it is a target extractor).

**Run + eval (fixed protocol, unchanged tooling).** Per arm:
`CUDA_VISIBLE_DEVICES=0,1,2 FORCE_TORCHRUN=1 scripts/train.sh configs/ad_bev_e2_3_lambdaX_seedY.yaml
deepspeed=examples/deepspeed/ds_z2_config.json --eval local_data/e2_overfit_lambda/heldout_infer.jsonl`
— ZeRO-2 is mandatory (3B full-finetune DDP OOMs, learned in E1). `train.sh` already produces the
timestamped run dir, `run.log`, 3-loss plot, and the held-out open-loop L2 in the same dir. The
`base` reference is the same `--eval` path pointed at `checkpoints/deepsight_warmstart` with no
training (or `src/infer_local_multi_gpu.py` directly).

**Resource / time estimate.** Same data and schedule as E2 (2000×2 epochs on 3 GPUs), so each arm
≈ one E2 arm's wall-time; 4 training arms + base eval fit the same ≤6 h budget E2 used. Disk:
`make_warmstart_init.py` writes one ~7 GB checkpoint (`deepsight_warmstart`) reused read-only by
all arms; per-run saves are the final model only (`save_strategy: "no"` keeps rolling ckpts off).

**Decision rule (pre-registered, so we don't post-hoc rationalize).** Report mean±range of
held-out L2 over seeds for `base`, `λ0`, `λ2`.
- **World loss helps** ⇔ `λ2 < λ0` by a margin **larger than the seed spread**, *and* `λ2 < base`
  (it must improve over the untrained start, not merely "degrade less" — the exact hole E2-2 had).
- `λ0 ≈ λ2` within seed noise ⇒ **no measurable benefit at this scale** (honest null; still a
  valid, comparable result — unlike E2-2).
- `λ2 > base` (both arms fail to beat the untrained model) ⇒ the **2000-sample budget is too
  small** to learn the task; revisit data size before concluding anything about the world loss.

**Carry-over caveats** (from the E2′ analysis, unchanged): this tests the world loss when
*fine-tuning a pretrained VLM on a small set*, not at DeepSight's full pretraining scale; a real
benefit may only surface with more data. Stated, accepted.

**Status: init built; ⚠️ TRAINING REGIME REVISED — the `finetuning_type: full` /
`freeze_vision_tower: false` choice above is SUPERSEDED by the LoRA decision below.** The
*init* (warm-start) and the *ablation logic* (neutral init, toggle only λ, base reference,
multi-seed, the decision rule) all stand unchanged; only **how we train on top of that init**
changed. Build progress so far: base Qwen2.5-VL-3B downloaded → `checkpoints/Qwen2.5-VL-3B`;
`scripts/make_warmstart_init.py` written & run → `checkpoints/deepsight_warmstart` (+ a seed-1
init was *not* needed — seeds only vary data order, init is fixed); smoke-tested (loads on
multi-GPU, generates; untrained `base` anchor is appropriately weak). The four full-FT configs
were a first cut; they are replaced per the regime decision below.

### E2-3 training regime: LoRA, not full fine-tune (final decision to implement)

**The decision.** E2-3 trains a **LoRA** adapter on the LLM trunk with the world head and the
new-token rows fully trainable, the Qwen ViT and DINOv3 frozen — *not* full fine-tuning.

**Why (judged against E2's actual goal, not paper faithfulness).** E2 exists to give a
**reliable, reproducible testbed that isolates the world head's marginal effect** and is reused
to compare *future* world-head designs (JEPA target, predictors, action-conditioning). The right
metric is therefore **signal-to-noise on the head's contribution, at low cost, held fixed across
variants** — not resemblance to the paper recipe. On that metric:
- **Reliability = effect ÷ noise.** Full-FT makes all 3.7B params plastic on only 2000 samples →
  the policy loss alone can fit the task, so the world head is one of two forces on a fully-moving
  trunk and its small contribution is buried in high seed variance (overfitting). LoRA pins the
  pretrained trunk and adds a regularized low-rank delta → it **cuts variance far more than it cuts
  the effect** → a true small effect becomes *detectable*.
- **Sensitivity to the head specifically.** With a fully-plastic trunk, any *future* world-head
  change washes out against the moving backbone. With the trunk pinned, the policy reads near-fixed
  base features + a small shared delta, so the world objective's reshaping of that shared substrate
  stays in sharp relief — the protocol remains attributable to the head across variants.
- **Reproducibility as a fixed protocol.** Future contributions are all changes to *how the world
  objective is computed*; everything else must be cheap to hold fixed and rerun. LoRA is
  single-GPU, fast, stable → the full multi-seed ablation can be re-run for each new head idea,
  apples-to-apples. Full-FT (ZeRO-2, multi-GPU, hours/arm) is too costly to be the recurring harness.

Full-FT on 2000 samples wins only "faithfulness," which is **not** E2's goal — and it buys no real
external validity anyway (1% of the paper's data with the paper's optimizer is cosmetic resemblance).
So full-FT-on-small-data loses on every axis that matters here; LoRA is the genuine choice, not a
cost compromise. The world head's gradient mechanism is preserved: `loss_gen` still flows into the
trunk via the LoRA delta (so the world objective shapes the representations the policy reads); a
*frozen* LLM would cut that pathway and was ruled out.

**Regime — what trains, what's frozen (the fixed E2-3 / future-head harness):**

| Component | Setting | Why |
|---|---|---|
| LLM trunk (`model.language_model.*`) | **LoRA**, rank **64**, `lora_target: all` | shared substrate the world loss shapes; low-variance, regularized |
| World head (`vis_head`) | **fully trainable** (`additional_target`) | the object under test — must have full capacity |
| New-token rows (`embed_tokens` / `lm_head`) | **fully trainable** (`additional_target`) | start random; must converge or both arms are null (E2-1 trap) |
| Qwen ViT (`model.visual.*`) | **frozen** | pinned substrate keeps the head's effect isolated; cuts params/noise |
| DINOv3 (`dinov3.*`) | **frozen** (forced in code) | target feature extractor only |
| Init | `checkpoints/deepsight_warmstart` | neutral-capable warm-start (unchanged from the design above) |
| `num_train_epochs` | **~5** (up from 2) | LoRA underfits faster — ensure the task is actually learned before reading the effect |
| `learning_rate` | **2.0e-4** | LoRA tolerates/needs a higher LR than full-FT |
| seeds | **≥2** (cheap now) | estimate the noise floor for the decision rule |
| λ_world arms | **{0, 2}** | the ablation variable |

**Unchanged from the design above:** neutral-capable warm-start init; identical setup across arms
toggling only λ (and seed); the `base` (no-train) reference; the **pre-registered decision rule**
(world loss helps ⇔ `λ2 < λ0` beyond seed spread *and* `λ2 < base`).

**Honest scope (this is a deliberate trade, not an oversight).** LoRA measures the head's effect
when adapting a *pinned* trunk — it does not predict the paper's full-FT-at-scale numbers. This
caveat applies to *any* small-scale probe (full-FT on 2000 included), so it is not a reason to
prefer full-FT; it is a scope statement. E2's purpose is a reliable comparator for world-head
designs, and this is exactly the harness every future world-head idea will be run through.

**One plumbing change vs the design above.** LoRA run dirs are *adapters*, so the auto-eval needs
a **merge-before-eval** step (`llamafactory-cli export` to fold the adapter + `additional_target`
modules into a full checkpoint, then the existing infer/eval runs unchanged on that merged dir).

**Status: regime decided — to implement.** Build order when greenlit: (1) rewrite the 4 configs
to `finetuning_type: lora` with the knobs above; (2) add the merge-before-eval step to
`scripts/train.sh`; (3) run the 4 arms + `base` eval; (4) tabulate vs the decision rule under
this day's log.
