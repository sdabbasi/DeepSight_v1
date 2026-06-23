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

### E2-3 training regime: LoRA, not full fine-tune (as run)

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

**Regime — what trains, what's frozen (the fixed E2-3 / future-head harness). The table below is
the AS-RUN config (`configs/ad_bev_e2_3_LORA_lambda{0,2}_seed{0,1}.yaml`); deltas from the
first sketch are flagged.** Trainable params: **779 M / 5.16 B = 15.1%** (verified at launch).

| Component | Setting | Why |
|---|---|---|
| LLM trunk (`model.language_model.*`) | **LoRA** rank **64**, alpha **128**, dropout **0.0**; `lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | shared substrate the world loss shapes; low-variance, regularized. **Changed from `lora_target: all`** — "all" would LoRA-wrap `vis_head`/`lm_head` (rank-capping a head that must train at full rank); explicit LLM-projection list avoids that. ViT is excluded anyway via `freeze_vision_tower`. |
| World head (`vis_head`) | **fully trainable** (`additional_target: vis_head`) | the object under test — must have full capacity |
| New-token rows (`embed_tokens` / `lm_head`) | **fully trainable** (`additional_target`) | start random; must converge or both arms are null (E2-1 trap) |
| Qwen ViT (`model.visual.*`) | **frozen** (`freeze_vision_tower: true`) | pinned substrate keeps the head's effect isolated; cuts params/noise |
| DINOv3 (`dinov3.*`) | **frozen** (forced in code) | target feature extractor only |
| Init | `checkpoints/deepsight_warmstart` | neutral-capable warm-start (built by `scripts/make_warmstart_init.py`) |
| Dataset | **`local_data/e2_lora` — 5000 train / 1000 eval, scene-disjoint** | **Changed from `e2_overfit_lambda` (2000/500).** On 2000 samples the arms converge within ~1 epoch and overfit (see 2026-06-17 entry), so a bigger, more diverse, scene-disjoint set was built (`scripts/build_e2_lora.py`, parallel) for a generalizable, lower-variance held-out comparison. |
| `num_train_epochs` | **3** | **Changed from ~5.** Smoothed train loss shows learning saturates ~epoch 1; 3 gives margin without heavy memorization. |
| `learning_rate` / scheduler | **2.0e-4**, `cosine`, `warmup_ratio 0.03` | LoRA tolerates/needs a higher LR than full-FT |
| batch | `per_device_train_batch_size 1`, `grad_accum 1` | keep effective batch constant across arms (same #GPUs) |
| dataloader | `num_workers 8`, `persistent_workers`, `prefetch_factor 4` | **Added** — with 0 workers the GPU starved ~50% waiting on NFS image decode (15 JPEGs/sample); prefetch overlaps decode with compute |
| `seed` | **{0, 1}** (≥2 seeds) | estimate the run-to-run noise floor for the decision rule |
| λ_world arms | **{0, 2}** (`world_loss_weight`) | the ablation variable |
| caching | `overwrite_cache: false` | arms share one tokenization cache → run #1 builds it, the rest reuse it (no concurrent-rebuild race) |

**Unchanged from the design above:** neutral-capable warm-start init; identical setup across arms
toggling only λ (and seed); the `base` (no-train) reference; the **pre-registered decision rule**
(world loss helps ⇔ `λ2 < λ0` beyond seed spread *and* `λ2 < base`).

**Honest scope (this is a deliberate trade, not an oversight).** LoRA measures the head's effect
when adapting a *pinned* trunk — it does not predict the paper's full-FT-at-scale numbers. This
caveat applies to *any* small-scale probe (full-FT on 2000 included), so it is not a reason to
prefer full-FT; it is a scope statement. E2's purpose is a reliable comparator for world-head
designs, and this is exactly the harness every future world-head idea will be run through.

**One plumbing change vs the design above — IMPLEMENTED.** LoRA run dirs are *adapters*, so the
auto-eval merges first: `scripts/train.sh` detects `finetuning_type: lora`, writes
`<run_dir>/export_config.yaml`, runs `llamafactory-cli export` to fold the adapter +
`additional_target` modules (`vis_head`/`embed_tokens`/`lm_head`) into `<run_dir>/merged/`, then
evaluates that. (Note: `export` only honours `key=value` overrides when arg 1 is a YAML file, so a
config file is passed — verified end-to-end; the merged ckpt contains the full DeepSight arch incl.
`dinov3`/`vis_head`.) Held-out open-loop L2 is computed on `local_data/e2_lora/heldout_lora.jsonl`.

**Status: IMPLEMENTED — sweep running.** Done: warm-start init built + smoke-verified; 4 LoRA
configs written; `train.sh` merge-before-eval added & verified; all three regimes (full-FT randinit,
full-FT warmstart, LoRA) smoke-passed end-to-end through `train.sh`; `e2_lora` (5k/1k) built. Run:
`scripts/train.sh configs/ad_bev_e2_3_LORA_lambda{0,2}_seed{0,1}.yaml --eval local_data/e2_lora/heldout_lora.jsonl`
(one GPU each; launch run #1 first so it builds the shared cache, then the rest). `base` reference =
`deepsight_warmstart` evaluated with no training. Results → tabulated under a later log entry vs the
pre-registered decision rule.

### 2026-06-17 — E2-3 implemented; fast-convergence finding drove the data scale-up

**Tooling implemented & verified.** Built the warm-start init (`scripts/make_warmstart_init.py` →
`checkpoints/deepsight_warmstart`: base Qwen2.5-VL-3B LLM+ViT, random `vis_head`/new-token rows,
frozen pretrained DINOv3). Made `scripts/train.sh` regime-aware (full-FT → eval run dir directly;
LoRA → merge adapter then eval). Smoke-tested all three regimes end-to-end through the one
`train.sh` (full-FT randinit + ZeRO-2; full-FT warmstart, λ=0 ⇒ `loss == loss_rec`; LoRA, merge→eval)
— all passed. Two bugs found & fixed: (a) **GPU starvation** — `dataloader_num_workers: 0` left the
GPU idle ~50% waiting on NFS decode of 15 JPEGs/sample; set workers=8 + prefetch across all configs.
(b) **LoRA merge** — `llamafactory-cli export` ignored bare `key=value` args (only honours overrides
when arg 1 is a YAML file), so `train.sh` now writes/export-passes a config file; verified the merged
ckpt carries the full arch (`dinov3`/`vis_head`).

**Fast-convergence finding (the reason data was scaled up).** Two LoRA arms on the 2000-sample
`e2_overfit_lambda` set ran 5 epochs (10000 steps). Smoothed training loss shows both arms do almost
all their learning in **epoch 1**, after which λ0 **plateaus** (memorizing 2000 samples) while λ2
keeps declining slowly (the `loss_gen` MSE target stays informative). On so few samples the held-out
L2 would be overfit-dominated and high-variance — a poor basis for a reliable world-loss comparison.

**Decision (taken with the user).** Increase data rather than just cut epochs — decreasing epochs
alone doesn't fix small-data overfitting/variance, whereas more diverse data does (and is exactly
E2's goal: a reliable, low-variance comparator). Built `local_data/e2_lora` via a new, saved,
**parallel** `scripts/build_e2_lora.py` (32-worker pool; ~3–4 min vs ~2 h serial — the work is
NFS-stat-latency-bound): **5000 train / 1000 eval**, scene-disjoint, excluding all 155
`e2_overfit_lambda` scenes, with asserted zero scene/sample overlap. Repointed the LoRA configs to
it, set `num_train_epochs: 3`, `overwrite_cache: false` (shared cache, run #1 builds it). The
2000-sample `e2_overfit_lambda` runs were the pilot that surfaced this; the reported E2-3 numbers
will come from the `e2_lora` sweep.

**Carry-over scope.** Still a small-data, pinned-trunk LoRA probe (not the paper's full-FT scale) —
its job is a reliable *relative* world-loss comparison and a fixed harness for future world-head
variants, not absolute paper L2.

### E2-3 results & conclusion: INCONCLUSIVE — no measurable world-loss benefit (and why that may be our design)

> ⚠️ These are the **pilot** results, run under a 2-split setup (train + a single held-out used
> as *test*), **no early stopping**, and L2 selected at the final step. They were the basis for the
> protocol upgrade below (3-split + periodic eval + early stopping). The upgraded sweep supersedes
> these numbers; conclusions here stand as the pilot read.

Four arms ran on `e2_lora` (5000 train / 1000 held-out, scene-disjoint; 2 epochs as-run, not the
configured 3; ~5.1 h each; held-out open-loop L2, 100% parsed). Overall L2: λ0 seed0 **1.369**,
λ0 seed1 **0.974**, λ2 seed0 **0.956**, λ2 seed1 **0.980**. Group means: **λ0 = 1.172 (seed spread
0.396)**, **λ2 = 0.968 (seed spread 0.024)**.

**Verdict against the pre-registered rule** (*helps ⇔ λ2 < λ0 by more than the seed spread, and
λ2 < base*): **FAILS.** Mean gap λ0−λ2 = 0.204 < λ0's seed spread 0.396; and **base was not run** (second
leg unverified). Reading the runs individually: **three of four cluster at ~0.95–0.98** — λ0 seed1
(0.974) sits right on the λ2 arms — and the only high value, λ0 seed0 (1.369), simply **converged
worse** (its train_loss 0.601 vs ~0.39–0.52 for the rest). So the apparent λ2 advantage is **driven
entirely by one unlucky λ0 seed**, exactly the spurious single-seed signal the multi-seed protocol
exists to reject (the E2-2 over-claim). **The protocol worked; the data is consistent with no
measurable world-loss benefit in this regime.** (Secondary, *not* a conclusion: λ2's spread is far
tighter than λ0's — the world loss *might* regularize training — but n=2 with one outlier is far too
little to claim it.)

### Critical: the paper claims the world head helps *remarkably*; we can't show it — what's wrong with OUR design?

A null here does **not** refute the paper. More likely, **our design answers a different question
than the paper's claim**, and several of our deliberate choices plausibly *removed the effect before
it could appear*. In rough order of severity:

1. **Wrong metric for the claim.** The paper's "remarkable" gains are **closed-loop Bench2Drive**
   (Driving Score, Success Rate, Multi-Ability on long-tail/interactive scenarios). We measure
   **short-horizon (1–2 s) open-loop L2**, which is largely solvable from current visual features +
   ego kinematics *without* world modeling. The world head's value — anticipating scene evolution,
   handling interaction/long-tail — barely projects onto 1–2 s waypoint regression. We are measuring
   the axis least sensitive to what the world model buys.
2. **LoRA pins the trunk, capping the world loss's mechanism.** The world objective is supposed to
   help by *reshaping the LLM's internal representations*. We froze the trunk and let only a rank-64
   delta + heads move — so `loss_gen` can only reshape a low-rank slice of the representations the
   policy reads. We chose LoRA for *reliability/reproducibility*; that very choice may have designed
   away the effect. The paper **full-fine-tunes**, where the world loss reshapes the entire trunk.
   We optimized for a trustworthy measurement of a regime where the effect can't fully express.
3. **Scale.** Paper: 64×H20, batch 128, full Bench2Drive, full-FT. Us: 5000 samples, batch 1, 1 GPU,
   2 epochs, LoRA. Auxiliary-representation benefits typically **grow with data/'plasticity**; at our
   scale the model fits the easy waypoint task directly and the world loss is marginal.
4. **In-distribution, easy held-out.** Train/eval are scene-disjoint but same simulator/Towns/scenario
   types. The world model is meant to pay off on **novel/long-tail/interactive** situations; an
   in-distribution short-horizon eval doesn't stress the regime where it should matter — and both arms
   saturate to a similar floor, leaving no room to separate them.
5. **No internal check that the objective did anything.** `loss_gen` decreased, but we never probed
   whether λ2's representations are actually more dynamics-aware (vs the head learning a degenerate
   solution that doesn't transfer). A null in the policy metric with no representation diagnostic
   can't distinguish "world loss useless" from "world loss worked internally but our metric/regime
   can't see it."

**Honest framing.** We can't currently separate two explanations — (a) **our probe is too weak/wrong
to surface a real effect**, or (b) **the paper over-states the benefit** (or it's entangled with
other components / only emerges closed-loop at scale). Our experiment, as built, is informative about
neither, because we traded the conditions the claim lives in (full-FT, scale, closed-loop, long-tail)
for conditions that are cheap and reliable (LoRA, small in-distribution data, open-loop L2). The
reliability we gained is real, but it was bought against a regime where the world loss has little to
do — so a clean null was, in part, **self-inflicted**.

**What would actually test the claim** (in increasing fidelity): (i) run the **base** reference to
complete the rule + add seeds 2–3 to settle the λ0 variance; (ii) **unfreeze the trunk** (full-FT or
much higher LoRA rank / LoRA on the whole stack) so the world loss can reshape representations;
(iii) move to a **longer-horizon / harder, more long-tail** eval split; (iv) ultimately, the only
faithful test is **closed-loop Bench2Drive at (something closer to) the paper's scale + full-FT** —
which our compute can't reach, so any small-scale verdict must stay scoped to "no *measurable*
open-loop benefit in a small pinned-trunk probe," not "the world head doesn't help."

### E2-3 protocol upgrade: 3-split data, periodic eval + early stopping, train-vs-eval diagnostic, FT mirror

Two diagnostics motivated this upgrade:

- **Fast convergence is not memorization.** Smoothed train loss does ~all its drop in epoch 1 then
  plateaus; epoch-2 loss is only marginally below epoch-1 at the same offset (no sharp epoch-boundary
  drop) — so it is the model hitting the task ceiling fast, not runaway overfitting. Cause: the task
  is **easy/low-entropy** (templated answer + near-straight waypoints) on a **strong pretrained
  warm-start** → transfer fits with little data. *Ruled out:* LoRA capacity (that would *under*fit,
  i.e. high train loss — opposite of observed); model size is minor (only 15% low-rank params train).
- **The world target is weak, not trivial (diagnostic on 128 held-out, faithful DINOv3 targets).**
  A scene-agnostic per-position-mean predictor scores MSE 0.0371; the model reaches 0.027 — so it
  *does* use scene info, but beats the floor by only **~27% of the scene variance**. DINOv3 features
  are small (std 0.28), so the "tiny" `loss_gen≈0.03` is largely a **scale artifact** (relative
  RMSE/std ≈ 0.6), and `loss_gen` saturates within ~100 steps → little sustained gradient. ⇒ the
  world objective is faintly informative; a future redesign should **normalize the target** and/or
  use a **harder (delta/contrastive/JEPA)** target. (My initial "trivial target" guess was *wrong* —
  recorded as such.)

**`eval_loss` vs L2 (why two metrics).** `eval_loss` = the *training* objective
(`loss_rec + λ·loss_gen`) on a held-out split via one **teacher-forced forward** (cheap, scores all
tokens incl. template). L2 = **autoregressive generation** + parse of the waypoints (expensive,
scores only the trajectory). They can disagree (teacher-forcing hides drift; template dominates
`eval_loss`). ⇒ use **`eval_loss` for early stopping**, **L2 on the untouched test set** for the
final verdict.

**Three scene-disjoint splits (verified train∩eval∩test = 0 at scene *and* sample level).** To stop
selecting on the test set (the pilot's flaw):
- `train_lora` 5000 (211 scenes) → fit; `eval_lora` 200 (43 scenes) → `eval_loss`/early stopping;
  `test_lora` 1000 (42 scenes) → final L2. Built reproducibly (`build_e2_lora.py`,
  `build_e2_lora_test.py`; both assert disjointness). `eval_lora` was trimmed to a 200-sample subset
  spanning all 43 eval scenes so each eval (forward pass) stays ~4 min instead of ~20 min at 1000.
- `local_data/e2_FT/` = a copy with `*_FT` filenames, so **FT trains on identical data** → FT-vs-LoRA
  is a controlled comparison (only the regime differs).

**Config changes (all 4 LoRA + all 4 FT arms).** `eval_dataset: bench2drive_bev_eval`,
`eval_strategy: steps`, `eval_steps: 500`, `per_device_eval_batch_size: 1`, `save_strategy: steps` +
`save_steps: 500` + `save_total_limit: 2` (required to align with) `load_best_model_at_end: true`,
`metric_for_best_model: eval_loss`, `greater_is_better: false`, `early_stopping_steps: 4` (patience),
`overwrite_cache: false` (shared cache; run #1 builds it). `train.sh` already merges the **best** LoRA
adapter before the final L2; for FT the best full checkpoint is evaluated directly.

**Early-stopping caveat (observed, not a bug).** LLaMA-Factory wires
`EarlyStoppingCallback(patience=early_stopping_steps)` but **does not expose
`early_stopping_threshold`** (defaults to 0). So *any* improvement — even 1e-4 — resets the patience
counter; a slowly-but-monotonically declining `eval_loss` therefore **never triggers** a stop. Seen
live (λ2 run): `eval_loss` set a new best at almost every eval (0.67→0.38 over 8.5k steps), so it ran
the full 2 epochs and `load_best_model_at_end` kept the best — correct behavior, just not "stop on
diminishing returns." To stop on a plateau we'd need to add an `early_stopping_threshold` knob
(small edit to `finetuning_args.py` + `tuner.py`) — deferred unless wanted.

**Over/under-fit diagnostic plot.** `scripts/plot_losses.py` now also emits **`train_vs_eval.png`**
from `trainer_log.jsonl`: total train loss (`loss_rec + λ·loss_gen`) overlaid with `eval_loss`, plus
an `eval − train` gap panel (flat = healthy, rising = overfit onset, both-flat = saturation). Built
from already-logged data, so it needs **no rerun**; `train.sh` regenerates it at the end of every run.
Mid-run snapshots of the live arms showed train≈eval with no widening gap → the saturation regime,
not overfitting.

**Operational note.** FT arms run multi-GPU (DDP/ZeRO), which **divides the optimizer-step count** by
#GPUs (data-parallel): e.g. 5000 samples × 2 epochs on 2 GPUs = **5000 steps**, not 10000 — same data
seen, fewer steps. Keep #GPUs consistent across arms being compared (effective batch must match).

**Status:** upgraded LoRA sweep running (with periodic eval + early stopping + 3-split); FT arms
configured identically on `e2_FT`. Final verdict to be tabulated (on `test_*`) once runs complete,
plus the still-pending **base reference** and extra seeds.

### 2026-06-18 — E2-3 results across BOTH regimes (FT and LoRA): no detectable world-loss benefit

First 2×2 of the upgraded protocol (regime × λ), **seed 0 only**, evaluated on the held-back
**test** split (1000 samples; `test_FT` and `test_lora` are identical content — `e2_FT` is a copy
of `e2_lora`). FT was 2 epochs on 2 GPUs (5000 steps); LoRA 2 epochs on 1 GPU (10000 steps); same
data exposure. (FT λ2 finished training but crashed in `load_best_model_at_end` — see the
save_total_limit note above — so its number is from the best+final `checkpoint-5000`, scored on
test; FT λ0 completed normally.)

| regime | λ | L2@1s | L2@2s | **L2 overall** | final eval_loss |
|---|---|---|---|---|---|
| FT   | 0 | 0.707 | 1.567 | **1.137** | 0.413 |
| FT   | 2 | 0.703 | 1.695 | **1.199** | 0.442 |
| LoRA | 0 | 0.551 | 1.305 | **0.928** | 0.318 |
| LoRA | 2 | 0.555 | 1.299 | **0.927** | 0.379 |

**Full exp titles (config → run dir, so the reader can reach them):**
- FT λ0 → `configs/ad_bev_e2_3_FT_lambda0_seed0.yaml` → `saves/ad_bev_e2_3_FT_lambda0_seed0/1781729898_ad_bev_e2_3_FT_lambda0_seed0/`
- FT λ2 → `configs/ad_bev_e2_3_FT_lambda2_seed0.yaml` → `saves/ad_bev_e2_3_FT_lambda2_seed0/1781733356_ad_bev_e2_3_FT_lambda2_seed0/`
- LoRA λ0 → `configs/ad_bev_e2_3_LORA_lambda0_seed0.yaml` → `saves/ad_bev_e2_3_LORA_lambda0_seed0/1781709395_ad_bev_e2_3_LORA_lambda0_seed0/`
- LoRA λ2 → `configs/ad_bev_e2_3_LORA_lambda2_seed0.yaml` → `saves/ad_bev_e2_3_LORA_lambda2_seed0/1781709872_ad_bev_e2_3_LORA_lambda2_seed0/`

**World-loss effect (λ2 vs λ0, within each regime — the comparable metric):**
- **LoRA:** 0.927 vs 0.928 → **identical** (Δ 0.001). No effect.
- **FT:** 1.199 vs 1.137 → λ2 **~5% worse**. No help; marginally negative.

⇒ **The world head improves open-loop L2 in *neither* regime.** Crucially this now holds in a
**fully-plastic FT** regime as well as the pinned-trunk LoRA one — which **addresses the earlier
objection** that LoRA was capping the world-loss→representation mechanism. Even with the whole trunk
trainable, turning the world loss on did not help.

**Caveats (do not over-read):**
- **Single seed each.** The earlier multi-seed run measured a λ0 **seed spread of 0.40**; the FT
  gap (0.06) and the LoRA gap (0.001) are both well inside that, so the FT "−5%" is **likely noise**,
  not evidence of harm. Honest claim: *no measurable positive effect.*
- **`eval_loss` is NOT comparable across λ** (λ2's includes the world-MSE term, λ0's does not) — only
  the waypoint **L2** is comparable across λ, which is what the verdict uses.
- Same standing scope limits: short-horizon **open-loop L2** is insensitive to what world-modeling
  should buy, and the world **target is weak** (~27% of scene variance; saturates in ~100 steps).

**Separate observation (orthogonal to the world loss):** **LoRA generalizes better than FT here**
(test L2 0.93 vs 1.14–1.20; eval_loss 0.32–0.38 vs 0.41–0.44) — the expected small-data signature
(full-FT of 3.7B on 5000 samples overfits; LoRA regularizes).

**Net for E2:** consistent **null** for the world head on open-loop driving, across two training
regimes — *not* proof it is useless (the faithful test — harder/normalized target, longer-horizon
or closed-loop, multi-seed — is still unrun). To make this publishable-grade: **add seeds 1–2 per
cell** (error bars on these gaps), run the **base/no-train reference**, and a **harder-slice** eval.

### Paper-faithful full-FT configs (FTpaper) + protocol hardening

The above used *our* recipe (lr 1e-4, trained vision tower, etc.), not the paper's. To remove the
"our hyperparameters were wrong" confound we built a **paper-faithful** full-FT family,
`configs/ad_bev_e2_3_FTpaper_lambda{0,2}_seed{0,1}.yaml`, after verifying the recipe against
`tex_source/`:

- **Paper hyperparameters (verified, and internally inconsistent):** main text
  (`sec/4experiments.tex`) says **lr 2e-5, batch 128**; the implementation paragraph
  (`main.tex:203`) says **lr 2e-4, batch 64** *and* — the key detail — **"the vision encoder is
  frozen, the LLM is fully fine-tuned."** Our earlier FT had `freeze_vision_tower: false`
  (unfaithful); FTpaper sets it **true**.
- **LR decision:** use **2e-5 regardless of batch size.** The two paper pairs contradict standard
  LR↔batch scaling (bigger batch → bigger LR, not 10× smaller), so they are not a principled pair;
  2e-5 is the standard/safe full-FT value, and 2e-4 is aggressive (instability risk). Batch kept at
  **64** via `gradient_accumulation_steps` (2 GPUs × 32). Consequence: 5000×2 epochs / 64 ≈ **158
  optimizer steps** — few, the faithful large-batch-on-small-data tradeoff.
- **Horizon = 2 s confirmed** (method §: Δt=0.5 s, 4 waypoints; world `F=[f0..f4]`=5 frames=2 s) —
  in both open- and closed-loop. So the **3–4 s eval idea was retracted** as un-faithful; eval stays
  1 s/2 s, matching the paper.
- **Init provenance (`checkpoints/deepsight_warmstart`):** LLM + vision = base Qwen2.5-VL-3B; new
  bev/pixel/CoT token rows + **`vis_head` = random**; DINOv3 = pretrained-frozen (Meta's, via the
  DeepSight ckpt). Nothing trainable is inherited from DeepSight. This is both **neutral** (fair
  ablation) and **faithful** (the paper also starts `vis_head` random and learns it during SFT;
  there is no separate trajectory head — waypoints are text via `lm_head`).
- **Early-stopping / crash hardening (all E2-3 configs):** the FT λ2 run crashed at end in
  `load_best_model_at_end` because `save_total_limit: 2` had deleted the tracked-best checkpoint.
  Fix: keep `load_best_model_at_end: true` but set **`save_total_limit ≥ patience+1`** (5 for
  patience 4, 4 for patience 3) — guarantees the best (which is ≤patience evals before the stop) is
  never rotated out, so load_best always finds it. (We briefly added an `early_stopping_threshold`
  knob to `tuner.py`/`finetuning_args.py`, then **reverted** it: stock threshold 0 is the *correct*
  default — early stopping is meant to fire on plateau/worsening, and our `eval_loss` was still
  improving, so it correctly never fired.)
- **Data factorization:** each dataset dir now holds `train.jsonl` / `eval.jsonl` / `test.jsonl`
  (type implied by the parent dir, e.g. `e2_lora`, `e2_FT`); `dataset_dir` alone selects all three
  (train+eval via the registry, test via `train.sh`'s `--eval` → `<dataset_dir>/test.jsonl`). The
  eval output was renamed `heldout_infer.json` → **`test_infer.json`**.

### 2026-06-19 — FTpaper results: the pipeline is VALID; the world head still shows no benefit

Ran the paper-faithful arms (seed 0 each; full-FT LLM, frozen vision, frozen DINOv3, lr 2e-5,
effective batch 64, 2 epochs = 158 steps; eval on the held-back `test` split, 1000 samples):
- λ0 → `configs/ad_bev_e2_3_FTpaper_lambda0_seed0.yaml` → `saves/ad_bev_e2_3_FTpaper_lambda0_seed0/1781793081_…/`
- λ2 → `configs/ad_bev_e2_3_FTpaper_lambda2_seed0.yaml` → `saves/ad_bev_e2_3_FTpaper_lambda2_seed0/1781798673_…/`

| arm | L2@1s | L2@2s | **L2 overall** | final eval_loss | train loss (first→last) |
|---|---|---|---|---|---|
| FTpaper λ0 | 0.609 | 1.438 | **1.024** | 0.347 | 4.83 → 0.285 |
| FTpaper λ2 | 0.702 | 1.584 | **1.143** | 0.461* | 36.4 → 0.395 |

*λ2 eval_loss includes the world-MSE term, so it is NOT comparable to λ0's — only L2 is.

**Did the pipeline work? YES.** With the faithful large-batch recipe the training is clean and
well-behaved: `eval_loss` falls **smoothly and monotonically** (λ0: 1.10→0.35; λ2: 1.51→0.46 over
the 7 evals), the model clearly learns (test L2 ≈ 1.0, in the same band as the other regimes), and
every component is wired (collator → BEV targets → `loss_gen` trains → `vis_head` predicts →
waypoints/CoT generate & parse 100% → test eval). So the **faithful, working small-scale pipeline
we were searching for is achieved.** (Minor: `eval_loss` is still inching down at step 140 → the
158-step/2-epoch budget is slightly short — the faithful large-batch-on-small-data consequence.)

**Did the world head help? NO — and now even in the faithful regime.** λ2 (1.143) is **~12% worse**
than λ0 (1.024) on test L2. Combined with the other regimes, the world loss never beats no-world-loss:

| regime | λ0 | λ2 | λ2 − λ0 |
|---|---|---|---|
| LoRA (e2_lora) | 0.928 | 0.927 | ~0 |
| FT, our recipe (e2_FT) | 1.137 | 1.199 | +0.06 (worse) |
| **FTpaper (e2_FT)** | **1.024** | **1.143** | **+0.12 (worse)** |

**Is this the valid setup we were searching for?** Two answers, and they differ:
- **As a *pipeline*: yes.** FTpaper is a faithful small-scale reproduction of the paper's *training*
  recipe (full-FT, frozen vision, frozen DINOv3, lr 2e-5, batch 64, 2 epochs, 2 s horizon), it runs
  cleanly, and it produces a sensible policy. It is a trustworthy testbed for future world-head ideas.
- **As a *demonstration that the world head helps*: no.** The setup removes the "wrong
  hyperparameters" excuse — and the null **persists** (slightly negative). So the paper's claimed
  world-head benefit does **not** surface in a faithful *small-scale, open-loop* probe.

**Honest caveats (so the negative isn't over-read):**
- **Single seed per arm.** The earlier multi-seed run showed a λ0 **seed spread ≈ 0.40**; the 0.12
  FTpaper gap is well inside that, so "λ2 worse" is **likely noise** — the defensible claim is *no
  measurable benefit*, not *harm*. Needs seeds 1–2 for error bars.
- **Slight undertraining** (eval still declining at 158 steps) and **no base/no-train reference** yet.
- The standing scope limit stands: the paper's "remarkable" gains are **closed-loop at full scale**;
  short-horizon **open-loop L2 at ~1% data** is the axis least sensitive to world modeling.

**Conclusion for E2.** The E2 program's *engineering* goal — a faithful, working, controlled
pipeline where each part is wired and the model trains — is **met** (FTpaper). Its *scientific*
question — does the world head help the policy at this scale — is a **consistent null across LoRA,
our-FT, and paper-faithful FT**. The world head's *mechanical* role works (it trains and predicts at
~27% of scene variance); its *functional* benefit to open-loop driving is not observable here. That
is now a clean, hyperparameter-confound-free statement, and it points the remaining explanation at
**scale / closed-loop / a stronger world target** rather than a broken setup. Next, to firm it:
multi-seed + base reference; then the decision of whether to pursue closed-loop/scale or a redesigned
(harder/normalized) world objective.

### FTpaper rerun at lr 2e-5 + 10 max-epochs + early stopping (the cleanest run; null holds)

After fixing the LR (2e-4 → **2e-5**, the safer/standard full-FT value; the 2e-4↔batch64 pair contradicts
LR-batch scaling) and raising `num_train_epochs` to **10** with early stopping, re-ran the FTpaper arms:
- λ0 → `configs/ad_bev_e2_3_FTpaper_lambda0_seed0.yaml` → `saves/ad_bev_e2_3_FTpaper_lambda0_seed0/1781858883_…/`
- λ2 → `configs/ad_bev_e2_3_FTpaper_lambda2_seed0.yaml` → `saves/ad_bev_e2_3_FTpaper_lambda2_seed0/1781858965_…/`

| arm | lr | test L2 overall | best eval_loss | stopped at |
|---|---|---|---|---|
| FTpaper λ0 (1781858883) | 2e-5 | **0.996** | 0.337 (ckpt-220) | early-stop @ step 280 / epoch 3.55 |
| FTpaper λ2 (1781858965) | 2e-5 | **1.006** | 0.427* | early-stop ~ same |

*λ2's eval_loss includes the world-MSE term → not comparable across λ; only L2 is.

**Two things this run validates (engineering goal):**
1. **Early stopping fired correctly.** With 10 epochs of headroom, λ0's `eval_loss` bottomed at step 220
   (0.3373) then rose for 3 evals (240→0.339, 260→0.347, 280→0.341) → patience-3 stop at 280, and
   **`load_best_model_at_end` restored `checkpoint-220` without crashing** — confirming the
   `save_total_limit ≥ patience+1` crash-guard end-to-end. (The earlier FT λ2 crash mode is fixed.)
2. The full protocol (3-split, periodic eval, early stop on best, full-FT direct eval) runs clean and the
   model generalizes (test L2 ≈ 1.0).

**World-loss effect (the cleanest comparison yet):** λ2 **1.006** vs λ0 **0.996** → **Δ +0.010 — essentially
tied** (λ2 a hair worse, deep inside the ~0.40 seed-spread noise). Compared to the lr-2e-4/2-epoch run
(λ0 1.024, λ2 1.143, Δ +0.119), the cleaner 2e-5 + more-epochs + best-model setup both **generalizes slightly
better** (λ0 1.024→0.996) and **shrinks the apparent λ gap to ~0** — i.e. the earlier "λ2 worse" was mostly the
aggressive LR / few steps, not the world loss. So in the most careful paper-faithful run, **λ2 ≈ λ0**, matching
LoRA (0.927 vs 0.928).

**Can we now conclude what E2 was searching for?**
- **Pipeline (engineering) — YES, conclusively.** We have a faithful, working, controlled small-scale
  reproduction of the paper's training pipeline: it trains cleanly, early-stops on the best checkpoint without
  crashing, generalizes, and every component is wired and exercised. This is the validated testbed E2 set out
  to build — future world-head ideas can now be A/B'd through it.
- **World-head benefit — NO measurable effect**, now a **consistent null across every regime tested**
  (LoRA Δ≈0; paper-faithful FT at 2e-5 Δ≈+0.01). It is *not* proof the head is useless: the open-loop /
  small-scale / short-horizon probe is the axis least sensitive to world modeling, and we still lack
  multi-seed error bars + a base reference. So: **"no measurable open-loop benefit at this scale," confirmed
  in the cleanest faithful setup** — the remaining honest explanations are scale / closed-loop / a stronger
  (harder/normalized) world target, not a broken pipeline.

**Docs updated alongside these runs** (so the mechanics are recorded once, in the right file): the
`loss_rec` (training CE) vs **L2** (eval) distinction, the forward/backward pass (hidden→`lm_head`/`vis_head`
projection, teacher forcing, the one-position label shift, what is/!isn't back-propagated), sequential
inference (prefill + decode loop, "only the last hidden row predicts the next token", KV-cache), and the
train/eval/test "is the GT fed to the model?" table were written into **`INPUT_FORMAT.md` §10** (verified on
the 4540-token sample). These were intentionally placed in `INPUT_FORMAT.md` (not `SRC_CODE_MAP.md`), which
remains the paper→code map.

---

### 2026-06-22 — world-head **feature collapse**: real in OUR repro, ABSENT in the released checkpoint

**How we got here.** Re-examining the two cleanest runs (`1781858883_…FTpaper_lambda0_seed0`,
`1781858965_…FTpaper_lambda2_seed0`) the loss curves looked *too* easy — eval_loss reaches ~90 % of its total
drop within the **first epoch** and is flat after ~1.9 epochs (eff-batch 64, 5000 samples → **78 steps/epoch**;
λ0 eval 0.43@step80→0.376@100→best 0.337@220 then rises = overfitting). That fast-convergence smell led to a
collapse investigation of the **world head** (`vis_head` + `loss_gen`), then to checking the paper, validating
our data, and finally probing the released checkpoint. Net result **reverses** the earlier "world loss does
nothing" reading and **retracts** a wrong claim made during this investigation (see Retraction below).

**TL;DR.** Our small-scale world head **collapses to the per-dimension mean** (`loss_gen` floor ≈ 0.042 ≈ the
trivial baseline). The **released DeepSight checkpoint does NOT collapse** (`loss_gen` = **0.019** on the *same*
data → ~54 % explained variance). So collapse is **not intrinsic to the unnormalized-MSE objective**; it is a
property of our **low-diversity / few-step** regime. E2's null was therefore a test of a *collapsed* head, not of
world modeling per se.

#### Part 1 — the DINOv3-target baseline table (what each row means, how it's computed)

`loss_gen = nn.MSELoss(vis_head(hidden@bev_positions), DINOv3(future_BEV))`, reduction = mean over **every**
element. The model's collator builds `template_mask` = per 261-token DINOv3 frame, **keep CLS (pos 0) + 256
patches (pos 5..260), drop the 4 register tokens (pos 1..4)** → 257 kept/frame. We ran the *released* model's
frozen DINOv3 on real `e2_FT` future-BEV crops (400 frames, registers dropped) and asked: *what MSE would a
**constant** predictor (one that ignores the scene) score, at three granularities?*

| Row | What the constant predictor may know | Computation | MSE |
|---|---|---|---|
| predict **global scalar mean** | one number for all 400×257×1024 elements | `c=F.mean(); ((F-c)**2).mean()` = total per-element variance (`std²=0.2675²`) | **0.0716** |
| predict **per-dim mean** | a fixed **vector** (1024), same for every scene & position | `m=F.reshape(-1,1024).mean(0); ((F-m)**2).mean()` | **0.0414** |
| predict **per-(position,dim) mean** | a fixed **257×1024 template**, indexed by token position, still scene-blind | `t=F.mean(0); ((F-t)**2).mean()` | **0.0320** |
| — achieved by **our FTpaper λ2** head | the actual trained model | training-log `loss_gen` plateau | **~0.042** |

Reading: our head sits **on the per-dim-mean baseline** (≈5 % explained variance) and is **worse than the
scene-blind per-position template** (0.042 > 0.032). A lookup table that ignores the input entirely would beat
it. That is the unambiguous collapse signature. (An earlier unmasked pass on the warmstart DINOv3 gave the same
story: 0.0734 / 0.0441 / 0.0308.)

#### Part 2 — the three arguments, the tests, and the verdicts

**Diagnostics on OUR runs.**
- **`loss_gen` collapses in <1 epoch and never recovers.** Windowed mean over the λ2 run's microbatch prints:
  16 (pre-warmup) → **0.064 by ~step 80 (1 epoch)** → crawls to **0.042** over the next ~9 epochs (last-200
  mean 0.045, std 0.005). It flatlines, it doesn't learn.
- **`vis_head` barely trains** (std of the 2048×1024 weight):

  | checkpoint | `vis_head.weight` std | note |
  |---|---|---|
  | warmstart (init) | 0.0200064 | random Linear (neutral ablation; **not** the pretrained head) |
  | λ0 trained (weight=0) | 0.0200064 | **byte-identical** to init (no gradient — expected) |
  | λ2 trained (weight=2) | 0.0199898 | moved **0.07 %** over 648 steps |
  | released `deepsight` | 0.0201 | different values — genuinely trained |

  The loss is "minimized" through the LLM trunk emitting near-constant hidden states at bev positions, not by a
  meaningful projection.

**The paper (tex_source).**
- **No mention of collapse / normalization / stop-grad / centering / variance / cosine** anywhere. `L_world =
  MSE(F, F_gt)` — same plain unnormalized MSE as the code; `λ_world` never even given numerically.
- **It does ablate the world model**, but compares *variants / on-off*, never *prediction quality*:
  - `tab:method_comparison` (220 routes, closed-loop): **WM off→on (ID1→ID3) = +26.4 DS, +37.7 SR**, called the
    biggest single contributor (bigger than CoT).
  - `tab:closed_loop_ablation` (Dev 10): DINOv3 target ≫ VAE (+47 DS); 5-frame ≫ 1-frame (+11.8 DS).
  - Crucially it **never reports `loss_gen`** or a feature-reconstruction metric — only that the *module's
    presence* helps. (Their WM toggle flips the whole 1305-token block + head + loss together, so it can't
    separate "predicting the future helps" from "extra register/compute tokens help".)

**User's three arguments — all VALID.**
1. *Eval-regime (closed vs open loop) can't explain no-collapse, since their **training** is also open-loop /
   teacher-forced / same MSE.* ✅ Collapse is a **training-time** property of the objective; the eval regime is
   irrelevant to whether the head collapses. The earlier "regime" reconciliation conflated *benefit-detectability*
   with *collapse-occurrence* — conceded.
2. *Huge drop in the first <1 epoch (before seeing each sample once) ⇒ data **quantity** isn't the cause.* ✅
   Collapsing to the mean is a low-complexity statistic reachable in a few batches; the fast drop **is** the
   collapse, not learning.
3. *Batch size / #GPUs can't be it — grad-accum matched effective batch 64.* ✅ Grad-accum reproduces the
   true batch-64 gradient exactly; there are no batch-coupled anti-collapse terms here (no contrastive negatives,
   no VICReg variance, no BN). Batch 64 vs the paper's 128 doesn't flip collapse.

**Data-equivalence validation (done BEFORE trusting the released-model probe).** Confirmed our preprocessing ==
theirs for the frames we use:

| Aspect | Ours | Theirs | Match |
|---|---|---|---|
| BEV crop geometry | `crop_bev_for_bench2drive_local.py` | `crop_bev_for_bench2drive.py` | **verbatim** (ego-motion warp, 512² crop @ top=85, hz=[0,5,10,15,20]); only I/O differs |
| Image list | 4 hist `rgb_front` + 6 surround + 5 BEV, BEV last | `create_date_set.get_images` (L144 → `rgb_{cam}`) | **identical** count/order/folders |
| Prompt text | `targetpointgen.get_prompt` | closed-loop agent `get_prompt` | **byte-identical** except CoT flag |
| Answer text | `targetpointgen.get_answer` (FLAGE=False) | same module | **identical** structure |
| BEV→DINOv3 norm | 256 resize + ImageNet mean/std (`ad_collator`) | same | **same** |

Two deliberate, non-distorting deltas: CoT flag `<CoT_flag_False>` (our no-CoT arm) vs agent's hardcoded `True`
(one token; world features predicted before CoT); and we start at frame ≥20 (real history) vs their
`hisblack.jpg` placeholder for frames 1–19 (we're a clean subset). The Chinese-CoT `create_date_set.py` is a
**separate/older** pipeline, *not* the released English-token format — we validated against the right one.

**Decisive test — released-checkpoint `loss_gen` probe.** `configs/probe_deepsight_released.yaml`: a
**forward-only** run (lr=0, max_steps=3, no eval/save) of `checkpoints/deepsight` through the *real* training
stack + `ADCollator` on `e2_FT`, so the per-microbatch `loss_gen` printed by `modeling_qwen2_5_vl.py` reflects
the released model **unchanged**. 42 microbatches: mean **0.0190**, std 0.0036, min 0.012, max 0.031.

| Model (same `e2_FT` data, same `loss_gen`) | `loss_gen` | expl. var vs per-dim (0.0414) | vs per-pos (0.0320) | verdict |
|---|---|---|---|---|
| per-dim mean (trivial) | 0.0414 | 0 % | — | baseline |
| per-position template (trivial) | 0.0320 | — | 0 % | baseline |
| **our small-scale FTpaper λ2** | **0.042** | **~5 %** | **−31 % (worse)** | **collapsed** |
| **released DeepSight** | **0.0190** | **54 %** | **41 %** | **healthy** |

The published head reaches **half the MSE of predicting the mean** and beats the per-position template — it
genuinely encodes scene structure. **Collapse is absent in their work.**

#### Retraction

Earlier in this investigation I argued the collapse was **intrinsic to the unnormalized-MSE objective** and that
"scale won't add an anti-collapse term." **That is wrong.** Same objective, same frozen DINOv3, no normalization
— and the released head learned fine (0.019). The collapse is specific to our **low-diversity, few-step**
small-scale regime.

#### Conclusions

- **Mechanism.** With low-diversity targets (our 5000 mostly-straight frames) the per-dim mean is a *good*
  solution → vanishing gradient → collapse by ~step 80. With rich/diverse targets (full Bench2Drive: turns,
  interactions, batch 128, ~2 epochs = thousands of steps) the mean leaves large residuals everywhere → strong
  persistent gradient → the head is forced to encode real structure. This **confirms the earlier "data
  *diversity*, not quantity" intuition** and refines it: it's diversity that prevents collapse, not raw count.
- **E2 reinterpreted.** The λ2≈λ0 null (LoRA Δ≈0; FTpaper Δ≈+0.01) was **not** a fair test of world modeling —
  it measured whether a **collapsed** head helps the policy (it can't). A valid test needs a **non-collapsed**
  head.
- **Paper's ablation credibility restored.** Their head demonstrably learned (~54 % explained variance), so the
  +26 DS from the WM toggle plausibly reflects real world-prediction rather than only extra register tokens —
  though their on/off toggle still can't fully separate the two, and they never report prediction quality.

#### Next steps (suggested, not yet run)

1. **Reproduce a non-collapsed head locally**: rebuild `e2` with **maneuver-balanced** scenes (turns / junctions
   / interactions over-sampled), train enough steps, and confirm `loss_gen` drops **below the per-position
   baseline (<0.032)** before re-running the λ0/λ2 A/B. Only then is the E2 world-loss question fairly answered.
2. **Add a collapse guard to the standard metrics**: log `loss_gen` **and** explained-variance-vs-mean during
   every run (a head at the per-dim baseline = collapsed) so we never again mistake collapse for "no benefit".
3. **Diagnostic eval that can see world modeling**: CE on **numeric waypoint tokens only**, and autoregressive
   L2 split by **horizon (1s/2s)** and **maneuver (straight vs turn)** — the open-loop teacher-forced eval_loss
   saturates on template + straight-line kinematics and is blind to the head.
4. This also strengthens the JEPA motivation ([WORLD_MODEL_JEPA.md](WORLD_MODEL_JEPA.md)): a
   normalized/variance-regularized target would make the head **collapse-resistant even in low-diversity
   regimes**, decoupling "does world modeling help" from "did we feed it diverse enough data".

#### Artifacts / repro

- `configs/probe_deepsight_released.yaml` — forward-only (lr=0) released-checkpoint `loss_gen` probe.
- Baseline computation (frozen DINOv3 on `e2_FT` BEV, exact register mask): per-dim 0.0414, per-pos 0.0320,
  global 0.0716; `std=0.2675`. (Released DINOv3; warmstart DINOv3 gave consistent 0.0441 / 0.0308 / 0.0734.)
- Storage hygiene (same day): deleted all intermediate `saves/*/*/checkpoint-*` (1.4 TB→157 GB); run-root
  best models kept. One casualty — the **crashed** FT λ2 run `1781733356` never consolidated a root model, so
  its weights are gone (results/logs preserved; it was the superseded non-paper 2e-4 variant).

---

### Cont. — E2-4 designed & implemented: detect + prevent world-head collapse

Direct follow-on to the collapse finding. Goal: (1) a permanent **collapse meter**, and (2) **objective-level
prevention** tested cleanly on the *same* collapse-inducing small data, so any lift is a design fix not a
data fix. Scope chosen: **A1 (cosine) + A2 (VICReg) on fixed data** (data-diversity arm A3 deferred).

**Component 1 — collapse meter (the "check").** `scripts/probe_world_collapse.py`: wraps the validated
forward-only probe (lr=0, `world_loss_type=mse` forced so it always reads RAW MSE), greps `loss_gen`, computes
the DINOv3 baselines with the exact register mask, and prints **EV = 1 − loss_gen / MSE_perdim** plus
EV-vs-per-pos. Verdict bands (from the 2026-06-22 measurements): **EV ≤ 10 % = COLLAPSED** (ours ~5 %),
**EV ≥ 40 % = HEALTHY** (released 54 %). Objective-agnostic, so A0/A1/A2 are directly comparable.

**Component 2 — prevention, config-driven.** New knob `world_loss_type ∈ {mse, cosine, vicreg}` (+
`world_var_coeff`, `world_cov_coeff`) wired exactly like `world_loss_weight`
(`finetuning_args.py` → `train/sft/workflow.py` → `model.config` → branch in `modeling_qwen2_5_vl.py` forward).
Formulations:
- **mse** (A0, control) — original plain MSE to raw DINOv3 (reproduces collapse).
- **cosine** (A1) — `mean(1 − cos(pred, target))` on unit-normalized rows; removes the magnitude/per-dim-mean
  trivial solution.
- **vicreg** (A2) — `MSE_invariance + var_coeff·variance_hinge + cov_coeff·covariance`. **variance hinge** =
  `mean_d relu(std_target[d] − std_pred[d])` (target std detached) — pushes each prediction dim's batch-std up
  to the frozen target's per-dim std; catches the "predict one constant vector" collapse even at `b=1` (the
  1285 bev-token rows per microbatch give the batch axis). **covariance** = mean squared **off-diagonal
  correlation** of standardized predictions ∈[0,1] (scale-free; decorrelates dims / fights dimensional collapse).

Configs: `configs/ad_bev_e2_4_{A0_mse,A1_cosine,A2_vicreg}_seed0.yaml` — clones of the collapsing FTpaper λ2
recipe (lr 2e-5, eff-batch 64, full-FT, vision+DINOv3 frozen, e2_FT, warmstart), differing **only** in
`world_loss_type`. A0 ≡ the existing run `1781858965` (can be reused/re-probed rather than rerun).

**Validation done (no full runs yet).**
- Args parse; loss-branch unit test on synthetic collapsed-vs-healthy predictions: mse 0.073/0.010,
  cosine 0.97/0.06, vicreg 0.34/0.01 — every objective penalizes the collapsed solution far more than MSE
  relative to a healthy one (cosine/vicreg create the steep escape gradient MSE lacks).
- **vicreg end-to-end smoke** (real model, 2 GPU ZeRO-2, max_steps=2): runs clean, `loss_gen ≈ 15.8` at init
  = the invariance MSE at random init (same as the plain-mse run's init), with var/cov bounded and small —
  they engage only as the model drives `std_pred→0`. Two scale bugs found & fixed first: raw covariance blew up
  (1120) → switched to **standardized correlation**; sum→**mean** off-diagonal (26→bounded).

**Decision rule for the runs.** Primary = **EV via the meter** (did the arm escape collapse: EV well above the
~5 % A0 floor, ideally ≥ per-pos i.e. beating 0.032). Secondary = **open-loop L2 λ0 vs λ2 under a non-collapsed
head** — the first *fair* test of "does world modeling help the policy," now possible because the head learns.

**Status:** infra complete & validated; the three arms are **not yet trained**. Next: run A1+A2 (reuse A0 from
`1781858965`), then `probe_world_collapse.py` on each, tabulate EV + L2.

---

### 2026-06-23 — E2-4 first results: cosine improves L2; the MSE-EV meter is blind to non-MSE heads

Ran the two prevention arms on the fixed collapsing data (full-FT, lr 2e-5, eff-batch 64, warmstart, e2_FT,
early-stopped ~epoch 4.5 on best ckpt): **A1 cosine** = `1782155340_…A1_cosine_seed0`, **A2 vicreg** =
`1782155405_…A2_vicreg_seed0`. **A0 = the existing mse run `1781858965`** (not retrained — A0's config is the
old FTpaper λ2 recipe verbatim plus an explicit `world_loss_type: mse`, which is the default, so re-running it
only reproduces the known collapse). Pre-flight: full train→save→load→generate→L2 pipeline validated end-to-end
on the vicreg path, and the shared `modeling_qwen2_5_vl.py` change confirmed byte-identical for the `mse`
default (A0 2-step forward → `loss_gen` 15.81/16.0 = original init), so old configs are unaffected; eval never
runs the loss branch (`labels=None`).

**Open-loop L2 (valid + cross-comparable; same 1000-sample test + eval_l2):**

| Arm | 1s | 2s | overall | vs A0 |
|---|---|---|---|---|
| λ0 (no world loss, `1781858883`) | 0.594 | 1.398 | 0.996 | −0.010 |
| **A0 mse / control (`1781858965`)** | 0.595 | 1.418 | **1.006** | — |
| **A1 cosine** | 0.562 | 1.302 | **0.932** | **−0.074** |
| **A2 vicreg** | 0.590 | 1.363 | **0.976** | −0.030 |

A1 cosine improves **both** horizons (clearest gain, ~1.5× the ~0.05 sample-mean SE → plausibly real); A2 vicreg
is marginal (within noise). Single seed ⇒ suggestive, not conclusive. NB: cross-arm `eval_loss` is NOT
comparable (A1 ~0.84, A2 ~0.48, A0 ~0.34) because `eval_loss = loss_rec + 2·loss_gen` and `loss_gen` is in
different units per objective — only L2 and a scale-invariant collapse metric compare across arms.

**Collapse meter (`probe_world_collapse.py`, raw-MSE EV vs per-dim baseline 0.0414) — and its blind spot:**

| Arm | raw MSE `loss_gen` | EV vs per-dim | meter says |
|---|---|---|---|
| A0 mse | 0.042 | ~5 % | collapsed (confirmed earlier) |
| A1 cosine | **11.1** | −26743 % | "collapsed" — **INVALID metric** |
| A2 vicreg | **0.057** | −38 % | "collapsed" — **misleading** |

**Key finding: the MSE-EV meter is the wrong instrument for non-MSE heads.**
- **A1 cosine — meter invalid.** Cosine loss is **scale-free** (constrains direction only), so prediction
  magnitude is unconstrained; raw MSE (11.1) just measures that free scale, not collapse. EV is meaningless here.
- **A2 vicreg — meter misleading.** MSE 0.057 is **above** the per-dim mean floor (0.0414) and the control's
  0.042, which means vicreg did **not** collapse to the constant mean (the variance term *did* add spread) — but
  the spread is **MSE-misaligned** with the targets (invariance term lost to the var/cov terms as tuned). Whether
  the *direction* tracks the scene, raw MSE cannot say.

So **"is there collapse?" is answerable only for A0 (yes); for A1/A2 the current meter cannot decide** — a real
methodological result: **the collapse check must be objective-matched (scale-invariant).** The provisional read
is A1 cosine = best L2 and the most promising, A2 vicreg = added (wrong) variance with no L2 payoff at these
coefficients (`var=1.0, cov=0.04` may over-weight variance vs invariance).

**Next steps.**
1. **Build a scale-invariant collapse diagnostic** and re-judge A1/A2: cosine-EV (1 − cos_loss / mean-direction
   baseline) and **cross-scene CKA** (representation similarity, scale/rotation-invariant), via an embedding
   dump of `vis_head` predictions + DINOv3 targets. Only this can confirm whether cosine/vicreg escaped collapse.
2. If A1 cosine is confirmed non-collapsed + better L2 → that's the first evidence a *non-collapsed* world head
   helps the policy at our scale (the real E2 question). Then add seeds for error bars.
3. Re-tune vicreg (raise invariance weight / lower `world_var_coeff`) so variance is target-aligned, not free.
