# DeepSight — Paper-to-Code Investigation Report

This document maps the DeepSight paper (ICML 2026 submission, sources in [tex_source/](tex_source/))
to the concrete implementation in this repository. It is intended as a reference for navigating
the codebase: "the paper says X — which file implements X?"

> **Paper title:** *DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving*
> **Base codebase:** a fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with a patched
> Qwen2.5-VL model and a vendored copy of HuggingFace `transformers` + Meta's DINOv3.

---

## 1. One-paragraph summary of the system

DeepSight is a unified generative-understanding VLM (denoted $M_{\text{uni}}$ in the paper) built on
**Qwen2.5-VL-3B**. From multi-view + historical camera frames it jointly produces, in a *single forward
pass*: (a) **latent BEV world features** $\mathbf{F}=[f_0..f_4]$ for the next 5 future frames (2 s ahead),
(b) an **adaptive Chain-of-Thought** text $T_{\text{cot}}$, and (c) **trajectory waypoints** $\mathbf{P}_t$.
The world features are supervised by aligning the model's hidden states (projected through a small head)
to **DINOv3** features extracted from ground-truth future BEV images, via an MSE "world loss". Trajectory
and CoT are ordinary text tokens trained with cross-entropy. Closed-loop evaluation runs in CARLA via the
Bench2Drive framework.

**Reported results / scale (paper §4):** SOTA on the closed-loop Bench2Drive benchmark — official **220 short
routes / 44 interactive scenarios** (5 routes each); the five metrics are Driving Score (DS), Success Rate (SR),
Efficiency, Comfortness, Multi-Ability (ablations use Route Completion / Infraction Score / DS). Open-loop **L2 = 0.58**.
Trained on **64× H20** GPUs, bs 128, lr 2e-5, 2 epochs (main text; Appendix lists lr 2e-4, bs 64).

---

## 2. The core mechanism in code

The single most important file is the patched Qwen2.5-VL model:

[src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)

In `Qwen2_5_VLForConditionalGeneration.__init__` ([:1377](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1377)):

```python
dinov3_config = DINOv3ViTConfig.from_json_file(config.dinov3_config)
self.dinov3 = DINOv3ViTModel._from_config(dinov3_config)
self.dinov3.requires_grad_(False)          # frozen feature extractor  (phi_dino)
self.vis_head = nn.Linear(hidden_size, 1024, bias=False)  # maps LLM hidden -> DINOv3 dim
self.loss_gen = nn.MSELoss()               # L_world
```

In `forward` ([:1520-1545](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1520)):

```python
if pixel_values_bevs is not None:                     # future BEV RGB images
    with torch.no_grad():
        target_embeds = self.dinov3(...).last_hidden_state   # F_gt = phi_dino(I_bev)  (Eq. 3)
        target_embeds = target_embeds[bevs_masks]
...
vis_hidden_states = hidden_states[label_bev_masks]    # hidden states at the bev_token positions
vis_embeds = self.vis_head(vis_hidden_states)         # predicted F
...
loss_rec = self.loss_function(logits, labels, ...)    # CE over text = L_traj + L_cot
loss_gen = self.loss_gen(target_embeds, vis_embeds)   # L_world = MSE(F, F_gt)
loss = loss_rec + 2*loss_gen
```

This is the literal implementation of the paper's **Driving-World Model** (§3.2) and **Unified Training
Strategy** (§3.3, Eq. 5).

> ⚠️ **Discrepancies worth knowing (paper vs. code):**
> - The composite loss weight on the world term is **hard-coded to `2`** (`loss = loss_rec + 2*loss_gen`).
>   The paper's sensitivity analysis (Appendix, Table on $\lambda_{world}$) reports the **best value is 1.0**,
>   not 2.0. `loss_rec` lumps $L_{\text{traj}}$ and $L_{\text{cot}}$ together (both are plain text tokens),
>   i.e. $\lambda_{\text{traj}}=\lambda_{\text{cot}}=1$ implicitly.
> - The paper main text uses **Qwen2.5-VL-3B**. The old `CLAUDE.md` claimed 7B — that was wrong; the
>   weight-merge tooling and agent target a 3B checkpoint.

---

## 2.5 Token I/O flow: where vision + world + text tokens fuse, and where the output splits

This section answers three precise questions:
1. *Where does the VLM accept vision, world-model and text tokens at the same time, and how are they
   prepared/concatenated?*
2. *Where is the VLM output received, and how is it divided to feed the heads?*
3. *Which code consumes which token type as the world-model / CoT / trajectory output?*

### 2.5.1 Input side — building ONE fused embedding sequence

All token *streams* become a single `inputs_embeds` tensor inside
[`Qwen2_5_VLModel.forward`](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1225)
(the inner model, not the `...ForConditionalGeneration` wrapper). Three kinds of tokens coexist in the
**same** sequence:

| Token stream | Token IDs in `input_ids` | How its embedding is produced |
|--------------|--------------------------|-------------------------------|
| **Text** (prompt, CoT `<think>…</think>`, waypoint text) | normal vocab IDs | embedding table lookup |
| **World-model / "World Queries"** `<|bev_token_i|>`, `<|start/end_bev_token|>`, and action `<|pixel_token_N|>` | added-vocab special IDs (baked into the checkpoint tokenizer) | embedding table lookup — they are **learnable token embeddings**, this is what realizes $\mathbf{Q}_{\text{world}}$ |
| **Vision** (4 history + 6 surround frames) | repeated `image_token_id` placeholders | run through the Qwen ViT, then **scattered into** the placeholder slots |

The fusion happens in three steps ([:1262-1314](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1262)):

```python
# (1) embed EVERY token id — text, <|bev_token_i|>, <|pixel_token_N|>, and image placeholders alike
inputs_embeds = self.get_input_embeddings()(input_ids)            # :1262-1263

# (2) encode the raw images with the Qwen vision transformer
image_embeds = self.get_image_features(pixel_values, image_grid_thw)   # :1266  (self.visual ViT)

# (3) overwrite ONLY the image-placeholder positions with the ViT features
image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)  # :1268
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)                            # :1271
```

**Line 1271 (`masked_scatter`) is the concatenation/fusion point.** After it, `inputs_embeds` is one
sequence holding vision features (at image slots), learnable world-query embeddings (at `<|bev_token_i|>`
slots), and text embeddings — all in token order as laid out by the prompt+answer template. That single
tensor is fed to the decoder stack at
[:1314 `self.language_model(..., inputs_embeds=inputs_embeds)`](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1314).
Self-attention across this fused sequence is exactly the paper's "deep self-attention" where
$\mathcal{X}$ and $\mathbf{Q}_{\text{world}}$ "interact and fuse" (§3.5).

> Note: `<|bev_token_i|>` are **not** replaced by anything (unlike image tokens) — they enter as plain
> learnable embeddings and the model is trained so their *output* hidden states carry the future latent.
> The token order itself (built in [create_date_set.py](src/tools/create_date_set.py) / `add_bev_text`)
> is what concatenates the three streams; there is no separate concat module.

### 2.5.2 Output side — receiving hidden states and splitting them to the heads

Control returns to
[`Qwen2_5_VLForConditionalGeneration.forward`](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1419),
which receives the decoder output and splits it by **token position** into exactly **two** projection
heads ([:1529-1544](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1529)):

```python
hidden_states = outputs[0]                                   # :1529  full fused-sequence hidden states

# HEAD A — language head (vocab logits) over the WHOLE sequence
logits = self.lm_head(hidden_states[:, slice_indices, :])    # :1533

# HEAD B — world-model head, fed ONLY the bev-token positions selected by the boolean mask
vis_hidden_states = hidden_states[label_bev_masks]           # :1535  pick <|bev_token_i|> positions
vis_embeds       = self.vis_head(vis_hidden_states)          # :1536  -> 1024-d latent F
```

So **`hidden_states = outputs[0]` (line 1529) is the point where the VLM output is received**, and the
split is done by the `label_bev_masks` boolean index (built by the AD collator, §4): masked positions →
`vis_head` (world model); all positions → `lm_head` (text).

### 2.5.3 The "three heads" — what is really there

The paper speaks of three outputs (world features, CoT, trajectory). In code there are **only two
`nn.Module` heads**; CoT and trajectory share one head and are separated downstream by the answer
template, not by a distinct layer:

| Paper output | Backed by | Selected/located by | Loss |
|--------------|-----------|---------------------|------|
| **World-model head** ($\mathbf{F}$) | `self.vis_head` (`nn.Linear(hidden, 1024)`) [:1385](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1385) | `label_bev_masks` boolean mask over `<|bev_token_i|>` positions [:1535](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1535) | `loss_gen` = MSE vs DINOv3 [:1543](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1543) |
| **Trajectory "head"** ($\mathbf{P}_t$) | shared `self.lm_head` | `<answer> future pixel tokens …</answer>` + `<answer> future waypoints …</answer>` spans (action `<|pixel_token_N|>` + numeric text) | part of `loss_rec` (CE) |
| **Adaptive-CoT "head"** ($T_{\text{cot}}$) | shared `self.lm_head` | `<think> … </think>` span (`<think> None </think>` = $T_{\text{cot}}^{\emptyset}$) | part of `loss_rec` (CE) |

i.e. trajectory and CoT are **not** separate output layers — they are different *token sub-strings* in the
single autoregressive text stream, distinguished by:
- **token type**: action waypoints use the dedicated `<|pixel_token_N|>` vocab; CoT uses ordinary
  natural-language tokens;
- **template delimiters**: `<think>…</think>` vs the two `<answer>…</answer>` blocks
  (assembled in [create_date_set.py:44-52](src/tools/create_date_set.py#L44));
- **decode-time parsing**: the agent/eval splits them back out with string/regex parsing in
  [qwen_b2d_agent.py `decode_traj` :712-725](bench2drive/team_code/qwen_b2d_agent.py#L712) and
  [eval_and_visual.py `parse_answer` :59-73](src/tools/eval_and_visual.py#L59).

The only *architecturally* separate output path is the world-model latent (`vis_head`); everything textual
flows through `lm_head`.

> **Where the special tokens come from:** `<|bev_token_i|>`, `<|pixel_token_N|>`, `<|start/end_bev_token|>`
> are **not** registered by any code in this repo — they were added to the tokenizer (and the embedding
> table resized) when the checkpoint was built offline. The repo only ever *looks them up* via
> `tokenizer.convert_tokens_to_ids(...)` ([ad_collator.py:299-300](src/llamafactory/data/ad_collator.py#L299)).

---

## 3. What the "World Queries" $\mathbf{Q}_{\text{world}}$ actually are

The paper describes learnable World Queries $\mathbf{Q}_{\text{world}}=[q_0..q_4]$ that let the model
predict all future frames in parallel in one pass. In code these are **special placeholder tokens
`<|bev_token_i|>` pre-filled into the assistant response**, not a separate learned tensor module:

- Count: `t * (h*w/patch² + n_cls + n_register) = 5 * (256 + 1 + 4) = 1305` tokens
  (frames=5, 256×256 image, patch 16 → 16×16=256 patches, + 1 CLS + 4 register tokens per frame).
- They are emitted between `<|start_bev_token|>` and `<|end_bev_token|>`.
- The LLM hidden states at these positions ARE the predicted latent features; `vis_head` projects them to
  the 1024-dim DINOv3 space for the MSE alignment.

Defined identically in three places:
- training data: [src/tools/create_date_set.py:126-131](src/tools/create_date_set.py#L126)
- transformers debug inference: [src/infer_for_debug.py:43-51](src/infer_for_debug.py#L43)
- closed-loop CARLA agent: [bench2drive/team_code/qwen_b2d_agent.py:675-683](bench2drive/team_code/qwen_b2d_agent.py#L675)

Because the BEV tokens are pre-filled (prefix), the "parallel prediction in a single forward pass"
(paper §3.5 Inference) is realized as the prefill stage of generation; the model then autoregressively
emits CoT + waypoints after `<|end_bev_token|>`.

---

## 4. BEV supervision masking (the AD collator)

[src/llamafactory/data/ad_collator.py](src/llamafactory/data/ad_collator.py) — class `ADCollector`
([:222](src/llamafactory/data/ad_collator.py#L222)). This is the "removes CE loss on BEV tokens" piece:

- Loads the **last 5 images** of each sample as the future BEV targets, resizes/normalizes to 256×256
  ([:284-291](src/llamafactory/data/ad_collator.py#L284)) → `pixel_values_bevs`.
- Finds the `<|start_bev_token|>`..`<|end_bev_token|>` span and sets those label positions to
  `IGNORE_INDEX` so the language CE loss does **not** apply to BEV tokens
  ([:312-318](src/llamafactory/data/ad_collator.py#L312)).
- Builds `template_mask` / `label_bev_masks` / `bevs_masks` that select which token positions feed the
  MSE world loss. `template_mask[:, 1:5] = False` drops the 4 register tokens per frame from supervision.

Wired into training at [src/llamafactory/train/sft/workflow.py:68](src/llamafactory/train/sft/workflow.py#L68)
(`road_collator = ADCollector(...)`, passed as the trainer's `data_collator`).

---

## 5. DINOv3 feature extractor

The model integrated into Qwen lives at:
[src/transformers/src/transformers/models/dinov3_vit/](src/transformers/src/transformers/models/dinov3_vit/)
(`modeling_dinov3_vit.py`, `configuration_dinov3_vit.py`). Imported into the Qwen model via
`from ..dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel` ([modeling_qwen2_5_vl.py:48](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L48)).

Meta's full original DINOv3 research repo is also vendored at [src/dinov3/](src/dinov3/) (training configs,
hub, etc.) but the runtime path used by DeepSight is the HF-style `dinov3_vit` model above.

Paper spec (Appendix): DINOv3-ViT-L/16, ~300M params, hidden size **1024** (matches `vis_head`'s output
dim), patch 16, pretrained on LVD-1689M. Frozen during SFT (`requires_grad_(False)`).

---

## 6. Trajectory & action token encoding (paper §3.3)

Waypoints are encoded two ways in the assistant answer:
1. **Pixel/action tokens** `<|pixel_token_N|>` — waypoints quantized to BEV-grid pixel coordinates,
   giving the CE-trainable "action tokens" the paper describes.
2. **Plain numeric waypoints** `(x,y)` text.

Construction + projection math (world→ego→camera→pixel) is in
[src/tools/create_date_set.py:104-123](src/tools/create_date_set.py#L104). The assistant answer template:
[create_date_set.py:44-52](src/tools/create_date_set.py#L44):

```
<think> {cot} </think>
<|start_bev_token|>{bev_tokens}<|end_bev_token|>
<answer> These are the future pixel tokens: {pixel_tokens}. </answer>
<answer> These are the future waypoints: {waypoints}. </answer>
```

Decoding back to trajectories (eval / closed-loop) parses these with regex:
[src/tools/eval_and_visual.py:50-73](src/tools/eval_and_visual.py#L50) and
[bench2drive/team_code/qwen_b2d_agent.py:712-725](bench2drive/team_code/qwen_b2d_agent.py#L712).

---

## 7. Adaptive Chain-of-Thought (paper §3.3 "Adaptive CoT")

- The CoT is the `<think> ... </think>` block. When reasoning is **not** needed, the placeholder
  `<think> None </think>` is emitted — this is the paper's $T_{\text{cot}}^{\emptyset}$ placeholder
  (see [src/infer_for_debug.py:51](src/infer_for_debug.py#L51)).
- The annotation pipeline (Qwen3-VL-235B, "scene complexity → external knowledge → behavior") that the
  paper's Appendix A describes is reflected in the Chinese reasoning prompt embedded in
  [src/tools/create_date_set.py:151-188](src/tools/create_date_set.py#L151), and the
  CoT-targeted dataset variant [src/tools/create_date_set_target_need_to_cot.py](src/tools/create_date_set_target_need_to_cot.py).
- At inference the agent prefills only the BEV block and lets the model decide whether to generate
  reasoning text vs. the placeholder before the answer
  ([qwen_b2d_agent.py:682](bench2drive/team_code/qwen_b2d_agent.py#L682)).

---

## 8. Data preparation pipeline (Bench2Drive → sharegpt JSONL)

| Step | File | Role |
|------|------|------|
| Raw Bench2Drive → conversational/target-point samples | [bench2drive/dataprocess/targetpointgen.py](bench2drive/dataprocess/targetpointgen.py) | Upstream step that turns raw collected data into the conversational training format (variants: `targetpointgen2.py`, `targetpointgenvae.py`) |
| Render/crop future BEV target images from CARLA top-down cam | [src/tools/crop_bev_for_bench2drive.py](src/tools/crop_bev_for_bench2drive.py) | Produces the 5 future BEV RGB images used as DINOv3 targets (verify with `visual_for_bev.py`; sensitive to weather / tall buildings) |
| Build training samples (15 images, prompt, answer w/ bev+pixel+waypoint tokens) | [src/tools/create_date_set.py](src/tools/create_date_set.py) | Main dataset builder |
| CoT-conditioned variant | [src/tools/create_date_set_target_need_to_cot.py](src/tools/create_date_set_target_need_to_cot.py) | Adds adaptive-CoT targets |
| Generate adaptive-CoT annotations (Qwen3-VL API) | [bench2drive/dataprocess/jsonopenai.py](bench2drive/dataprocess/jsonopenai.py) | Calls the Qwen3-VL model (OpenAI-style API) to produce the `<think>…</think>` CoT content (Appendix A annotation pipeline) |
| Convert to Qwen Bench2Drive format | [src/tools/convert_to_qwen_b2d.py](src/tools/convert_to_qwen_b2d.py) | Format adapter (`load_bev_tokens`) |
| Dataset registry | [data/dataset_info.json](data/dataset_info.json) | `bench2drive_bev_{train,val,test}` → NAS JSONL paths, `sharegpt` format |

**15 images per sample** = 4 historical CAM_FRONT frames + 6 surround-view current frames + 5 future BEV
frames. Prompt structure in [create_date_set.py:34-41](src/tools/create_date_set.py#L34). The future BEV
frames are popped off the end of the image list by the collator (§4).

---

## 9. Inference / evaluation paths

| Path | File | Notes |
|------|------|-------|
| HF transformers debug inference | [src/infer_for_debug.py](src/infer_for_debug.py) | Prefills bev tokens, `model.generate`, decodes text |
| Open-loop eval + visualization | [src/tools/eval_and_visual.py](src/tools/eval_and_visual.py) | L2 / collision style metrics, draws trajectories |
| Eval entry (LLaMA-Factory style) | [src/eval.py](src/eval.py) / [src/eval.sh](src/eval.sh) | Builds dataset+ADCollector, runs model |
| Weight merge (strip dino/vis_head for vLLM) | [src/tools/merge_model_weight.py](src/tools/merge_model_weight.py) | Drops `dino*` and `vis_head*` keys so a vanilla Qwen2.5-VL can serve |
| Closed-loop CARLA agent | [bench2drive/team_code/qwen_b2d_agent.py](bench2drive/team_code/qwen_b2d_agent.py) | `QwenAgent`; PID controller turns waypoints into steer/throttle/brake |
| Closed-loop launcher | [bench2drive/leaderboard/scripts/run_evaluation_qwen.sh](bench2drive/leaderboard/scripts/run_evaluation_qwen.sh) | Needs separate Python 3.10 CARLA env (see `example.txt`) |

Note: `merge_model_weight.py` removing `dino`/`vis_head` confirms that at deployment the world-model head is
only needed for *training* the latent prediction; the trajectory output itself is plain text generation, so
a standard serving stack (vLLM) can run the merged weights — consistent with the paper's claim of
"no additional external generative models" at inference (§3.5).

---

## 10. Training launch

- [nebula.sh](nebula.sh): `llamafactory-cli train --config ./configs/ad_bev_v4.yaml`
  **or** `torchrun --nproc_per_node=8 src/train.py --config ./configs/ad_bev_v4.yaml`.
- ⚠️ **`configs/ad_bev_v4.yaml` is not present in the repo** (the `configs/` dir is empty). You must supply
  the training YAML (model path, dataset = `bench2drive_bev_train`, lr, batch size, epochs). Paper settings:
  64× H20 GPUs, 2 epochs; LR/batch differ between main text ($2\times10^{-5}$, bs 128) and Appendix
  ($2\times10^{-4}$, bs 64).
- SFT workflow that actually wires everything: [src/llamafactory/train/sft/workflow.py](src/llamafactory/train/sft/workflow.py).

---

## 11. Paper-section → code map (quick index)

| Paper location | Concept | Primary code |
|----------------|---------|--------------|
| §3.1 Preliminary, Eq.1 | $M_{\text{uni}}$ joint output F, T_cot, P | `Qwen2_5_VLForConditionalGeneration.forward` ([modeling_qwen2_5_vl.py:1421](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1421)) |
| §3.2 Driving-World Model, Eq.2 | parallel future latent prediction via World Queries | `<|bev_token_i|>` prefill + `vis_head` |
| §3.2 Eq.3 | $f_i = \phi_{dino}(I^{bev}_i)$ GT construction | `self.dinov3(pixel_values_bevs)` ([:1524](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1524)) |
| §3.3 Adaptive CoT, Eq.4 | `<think>…</think>` / `None` placeholder | data builders + agent prefill |
| §3.3 action tokens | `<|pixel_token_N|>` quantized waypoints | [create_date_set.py:104](src/tools/create_date_set.py#L104) |
| §3.3 Eq.5 composite loss | `loss = loss_rec + 2*loss_gen` | [modeling_qwen2_5_vl.py:1538-1544](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1538) |
| §3.3 $L_{world}$ = MSE | `self.loss_gen` MSELoss | [:1386](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1386), [:1543](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1543) |
| §3.3 BEV tokens excluded from CE | label span → IGNORE_INDEX | [ad_collator.py:312-318](src/llamafactory/data/ad_collator.py#L312) |
| §3.5 Inference (single pass), Eq.6 | prefill X+bev, generate cot+traj | [qwen_b2d_agent.py:727-745](bench2drive/team_code/qwen_b2d_agent.py#L727) |
| §4 Datasets / Bench2Drive | data prep + CARLA eval | `src/tools/*`, `bench2drive/` |
| §4 Open-loop nuScenes (Appendix) | L2 / collision visualization | [eval_and_visual.py](src/tools/eval_and_visual.py) |
| Appendix A annotation pipeline | Qwen3-VL-235B CoT prompt | [create_date_set.py:151-188](src/tools/create_date_set.py#L151) |
| Appendix Implementation Details | base model = Qwen2.5-VL-3B, frozen vision | merge tooling + frozen `dinov3` |

---

## 12. Caveats / things that are NOT in this repo

- The training YAML `configs/ad_bev_v4.yaml` (referenced by `nebula.sh`) is missing — in fact there is no `configs/` dir at all.
- `requirements.txt` is **absent** (not committed); `pip install -r requirements.txt` from the README will fail.
- **Stale README/`nebula.sh` pointers:** `src/train.py`, `src/infer_with_vllm.py`, and `src/utils/merge_model_weight.py` are referenced but **do not exist**. The real entry points are `llamafactory-cli train` (distributed launch handled internally by `src/llamafactory/launcher.py`), `scripts/vllm_infer.py` (stock LLaMA-Factory vLLM), and `src/tools/merge_model_weight.py`.
- All dataset / checkpoint paths point to internal NAS mounts (`/mnt/nas-data-1/...`) and are not bundled.
- `data/dataset_info.json` BEV entries point to NAS JSONL files not present locally.
- The DINOv3 checkpoint and `config.dinov3_config` JSON path are external; the model reads them at init.
