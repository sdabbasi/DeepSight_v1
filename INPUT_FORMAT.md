# DeepSight — Input Data Format (token types, fusion, attention)

A reference for how one sample (`local_data/infer_samples.jsonl` → the model) is
turned into tokens, how images are injected, and what actually flows through the
transformer. **All numbers below were verified live** against the released
checkpoint (`checkpoints/deepsight`) and one real sample, not from memory.

Model dims (from `config.json`): `hidden_size=2048`, `num_layers=36`,
`num_attention_heads=16`, `num_key_value_heads=2` (GQA), `head_dim=128`,
all layers `full_attention`, `tie_word_embeddings=true`. Vision tower:
`patch_size=14`, `spatial_merge_size=2`, `temporal_patch_size=2`,
`out_hidden_size=2048`, `image_token_id=151655`.

The worked example is line 0 of `infer_samples.jsonl`
(`AccidentTwoWays_Town12_Route1102_Weather10`, frame 20). It tokenizes to a
**4540-token** sequence: **2990 image + 1305 bev + 12 pixel + 233 text/control**.

---

## 1. The token families

| In the on-disk text | Real vocab token? | Count (this sample) | Token id(s) | Purpose |
|---|---|---|---|---|
| `<image>` | **No** — a literal marker string | 10 markers | — (expands at runtime, see §2) | one placeholder per camera image |
| `<\|image_pad\|>` | yes | 2990 (299 × 10) | `151655` | runtime image slots; **overwritten** by ViT features (§3) |
| `<\|start_bev_token\|>` / `<\|end_bev_token\|>` | yes | 1 + 1 | `151669` / `151670` | delimiters of the world-model block |
| `<\|bev_token_i\|>` (i = 0…1304) | yes, learnable | 1305 | `151671 + i` → `151671…152975` | the **World Queries**: output slots for the future-BEV latent |
| `<\|pixel_token_N\|>` (N = −255…255) | yes, learnable | 12 (6 route pts × 2) | `153231 + N` → `152976…153486` (511 tokens) | a quantized BEV-grid coordinate (one integer) |
| ordinary text / numbers | yes | rest | normal vocab | prompt, `<think>`, waypoint digits, `<answer>`, `<CoT_flag_*>` |

Verified id formulas: `bev_token_i = 151671 + i`, `pixel_token_N = 153231 + N`.

### Where the token ids come from (why `<|image_pad|>` is 151655)
The id `151655` is **inherited from the base Qwen2.5-VL tokenizer** — DeepSight did
not choose it. The vocabulary is laid out as:
- **0 … 151642** — the 151,643 ordinary BPE text tokens.
- **151643 … 151664** — Qwen's built-in special tokens, in fixed order:
  `<|endoftext|>`(151643), `<|im_start|>`(151644), `<|im_end|>`(151645),
  object/box/quad markers (151646–651), `<|vision_start|>`(151652),
  `<|vision_end|>`(151653), `<|vision_pad|>`(151654), **`<|image_pad|>`(151655)**,
  `<|video_pad|>`(151656), tool-call / FIM / repo tokens (151657–664). So
  `<|image_pad|>` is simply the **13th** special token after the text vocab.
- **151665 …** — tokens DeepSight **appended** after all of the above:
  `<answer>`/`<think>` (151665–668), `<|start_bev_token|>`/`<|end_bev_token|>`
  (151669/670), the 1305 `<|bev_token_i|>` (151671–152975), the 511
  `<|pixel_token_N|>` (152976–153486). Total vocab = 153536.

So 151655 is not meaningful by itself; it's just where Qwen already placed its image
placeholder, and DeepSight's own tokens live in the higher appended range.

### Why those counts
- **bev = 1305** = `5 future frames × (256 patches + 1 CLS + 4 register)` = `5 × 261`.
  These are **constant in every sample** — pure query slots, carrying no input
  information themselves. Their *output* hidden states are the prediction (§7).
- **image = 299 per image** for this repo's resize (see §2/§3).
- **pixel = 511 possible values** but only a few used per sample — detailed next.

### Pixel (coordinate) tokens: range, meaning, and why they're learnable
- **How many appear: 12 in this inference input.** The prompt's `target pixel tokens`
  list holds **6 route points × 2 coords = 12** `<|pixel_token_N|>` (see §4). The
  ground-truth answer adds 8 more (4 future points × 2), but the answer is *generated*,
  not fed in. So **511 is the size of the coordinate sub-vocabulary** (possible values),
  while only ~12–20 are *used* in any one sample.
- **What the ±255 range means.** A point is projected into the **TOP-DOWN (BEV) camera**
  image (verified: `image_size = 1600×900`, principal point `cx,cy = 800,450`), then
  quantized to a grid centred on the ego:
  `dy = round((450 − v)/2)`, `dx = round((u − 800)/2)`, clamped to `[−255, 255]`.
  Each axis is therefore an **offset from the BEV image centre in units of 2 pixels**,
  giving `255 − (−255) + 1 = 511` cells per axis. **It is the BEV / top-down view, NOT
  the front camera.**
- **What a pair means.** `(<|pixel_token_21|>,<|pixel_token_0|>)` = `(dy=21, dx=0)` =
  **21 cells straight ahead, 0 lateral**. Order is `(forward, lateral)`: `dy>0` = forward
  (a point ahead of the ego is *above* centre in a top-down image, so `v<450`); `dx>0` =
  to the right; one cell ≈ 2 px of the 1600×900 BEV image. That is why this straight-
  driving sample's future pixels are `(21,0),(43,0),(65,0),(87,0)` — growing forward,
  zero lateral.
- **Why a coordinate token is a *trainable* vector (not a hardcoded pixel).** The integer
  `N` is only a **label / index** into the vocabulary — the model has no built-in
  arithmetic notion that id `153252` "means 21 cells forward." The `N → pixels` mapping is
  a fact about how the **data** was built (the projection formula above), not something
  baked into the token. For the network to *use* the symbol it must learn a 2048-dim
  vector for it, for two reasons:
  1. **As input** (the target route points): the embedding has to inject "a goal at offset
     N" into the 2048-dim space so other tokens can attend to it — a bare integer cannot
     participate in attention.
  2. **As output**: emitting a pixel token is a classification over the vocab, so the
     `lm_head` needs a learned weight vector per coordinate to score it. Because
     `tie_word_embeddings=true`, that output vector **is the same** learned vector as the
     input embedding — one shared vector per coordinate.

  Training these vectors lets the model capture structure a raw index can't: **ordinality
  / locality** (token 21 sits near 20 and 22), and the links between a pixel coordinate,
  the numeric **waypoint digits** in the answer, and the **BEV scene geometry**. (Same idea
  as the word "five" having a trainable embedding even though it denotes a fixed quantity:
  being a discrete symbol and having a learned vector are independent.)

---

## 2. `<image>` vs `<|image_pad|>` (on-disk vs runtime)

The JSONL only ever contains the marker **`<image>`** (you will *not* find
`<|image_pad|>` in `tmp.json`). The expansion happens inside the processor at
runtime:

```
on disk:            ... CAM_FRONT:<image> CAM_FRONT_LEFT:<image> ...
apply_chat_template:... CAM_FRONT:<|vision_start|><|image_pad|><|vision_end|> ...
processor(...):     ... <|vision_start|> <|image_pad|>×299 <|vision_end|> ...   → input_ids
```

**Mapping image → marker is by ORDER.** `format_message` splits the prompt on
`<image>` and interleaves `images[k]` for the k-th marker (images 0–3 = the 4
history `rgb_front`; 4–9 = the 6 surround cams). The processor then stamps out
the right number of `<|image_pad|>` per image, and in the model the k-th image's
feature vectors land in the k-th image's slots (§3). There is no explicit image
id — position is the binding.

---

## 3. From a 2D image to tokens (patchify) and the "scatter"

### Patchify — verified on this sample
`image_grid_thw = [1, 26, 46]`, `pixel_values.shape = (11960, 1176)`:
1. **Resize** each image to 364 × 644 (`resized_height/width` in `format_message`).
2. **Cut into 14×14 patches** → grid `26 × 46 = 1196` patches (364/14=26, 644/14=46).
3. **Flatten each patch** to `3 channels × 2 temporal × 14 × 14 = 1176` numbers.
   Stack: `(10 imgs × 1196, 1176) = (11960, 1176)`. **This is the 2D→sequence step**:
   the image becomes a list of patch-vectors.

   > **What is the "× 2 temporal"?** It is the vision tower's `temporal_patch_size = 2`,
   > a low-level packing detail of the Qwen ViT — **not** the 4 history frames. The
   > patch-embedding conv has a temporal kernel of 2, so it always consumes frames **in
   > pairs**; a single still image is **duplicated into 2 identical frames** so the conv
   > can apply, and that pair becomes **one** temporal patch — which is why
   > `image_grid_thw` has **t = 1** per image and each patch carries `3 × 2 × 14 × 14 = 1176`
   > numbers. The 4 history frames are a *different axis*: they are 4 **separate** images
   > (4 `<image>` markers → 4 separate 299-token blocks, §4). So temporality-across-time =
   > multiple images in the sequence; the "× 2" = each *one* image padded to the ViT's
   > 2-frame temporal patch.
4. **ViT + 2×2 spatial merge** → `1196 / 4 = 299` tokens per image, each **2048-dim**
   (`out_hidden_size`). 10 images → **2990** vectors of dim 2048.

### The scatter (`masked_scatter`)
"Scatter" is the PyTorch op that copies a source tensor into the positions of a
destination where a boolean mask is `True`, **in order**.
[modeling_qwen2_5_vl.py ~1262–1271](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1262):

```python
inputs_embeds = self.get_input_embeddings()(input_ids)            # embed EVERY id (text, bev, pixel, image_pad)
image_embeds  = self.get_image_features(pixel_values, grid_thw)   # ViT -> (2990, 2048)  ← "the vectors"
image_mask,_  = self.get_placeholder_mask(input_ids, ...)         # True at the 2990 <|image_pad|> rows
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)   # overwrite those rows
```

- **"The vectors that get scattered"** = `image_embeds`, the ViT output (2990 × 2048).
- The `<|image_pad|>` rows initially hold the *embedding-table vector for id 151655*
  — meaningless filler — and `masked_scatter` **overwrites** them with the real ViT
  vectors. Text / bev / pixel rows are untouched.

Tiny illustration (scalars for clarity):
```
dest   = [a, b, c, d, e]      # inputs_embeds rows
mask   = [F, T, T, F, T]      # True where id == <|image_pad|>
source = [x, y, z]            # the ViT vectors, in order
result = [a, x, y, d, z]      # poured into the True slots, left→right
```
**Legend (each letter = one sequence row = one token's 2048-vector, drawn as a single
symbol for readability):** `a,b,c,d,e` = the 5 rows already in `inputs_embeds` (in this
toy `b,c,e` happen to be `<|image_pad|>` slots, `a,d` are text); `F/T` = the boolean
mask; `x,y,z` = the 3 incoming ViT vectors. `masked_scatter` pours `x,y,z` into the
`True` rows **in order** → `b→x, c→y, e→z`; the text rows `a,d` are left untouched.

Because each image's pad-slots are contiguous and in image order, image-k's 299
vectors land exactly in image-k's 299 slots.

---

## 4. The fused sequence is "1D of vectors", not "1D of numbers"

After the embedding step, **every** position — text, image, bev, pixel — is a
2048-dim vector. So a single image's `(299, 2048)` is **not flattened**; it simply
occupies **299 consecutive rows** of the sequence, exactly like 299 words would.

The **exact** layout of the 4540 rows (verified by tokenizing the real inference
input = chat template + prompt + the prefilled BEV block; note the `<think>`/`<answer>`
answer is NOT in the input — the model *generates* it):

```
 rows  segment (→ each row is a 2048-vector)
 ----  -----------------------------------------------------------------
   31  <|im_start|>system … assistant.<|im_end|> … user\n + "…2.0s ago "   TEXT
  301  history frame 1  =  <|vision_start|>(1) + <|image_pad|>×299 + <|vision_end|>(1)
    7  " 1.5s ago "                                                        TEXT
  301  history frame 2   (rgb_front, 1.5s ago)
    7  " 1.0s ago "                                                        TEXT
  301  history frame 3
    7  " 0.5s ago "                                                        TEXT
   15  ".\nThese are the … six-view images: CAM_FRONT:"                    TEXT
  301  surround CAM_FRONT
    4  " CAM_FRONT_LEFT:"        + 301  surround CAM_FRONT_LEFT
    4  " CAM_FRONT_RIGHT:"       + 301  surround CAM_FRONT_RIGHT
    3  " CAM_BACK:"              + 301  surround CAM_BACK
    4  " CAM_BACK_LEFT:"         + 301  surround CAM_BACK_LEFT
    4  " CAM_BACK_RIGHT:"        + 301  surround CAM_BACK_RIGHT
    9  ".\nThese are the target pixel tokens: [("                          TEXT
   23  6 target route points = 12 PIXEL tokens + 11 punctuation TEXT tokens
  104  ")] Historical trajectory: […] speed:… <CoT_flag_False>\nBased on…
        next 2 seconds.\n<|im_start|>assistant\n"                          TEXT
    1  <|start_bev_token|>
 1305  <|bev_token_0 … 1304|>
    1  <|end_bev_token|>
    1  "\n"                                                                TEXT
 ----  -----------------------------------------------------------------
 4540  TOTAL
```

**Where 4540 comes from — and yes, it includes text tokens (211 of them):**
```
4540 = 2990 image_pad      (299 × 10 cameras)
     +   20 vision markers (<|vision_start|>+<|vision_end|>, 2 × 10)
     + 1305 bev queries
     +    2 start/end_bev markers
     +   12 pixel tokens    (6 target route points × 2 coords)
     +  211 text tokens     (chat template + camera-label text + route/speed/
                             instruction text + assistant header + trailing "\n")
```

"1D" = the **sequence/position axis** (length 4540). "2048" = the **per-token
feature axis**, identical for every token type. A 2D image was already linearized
into 299 patch-tokens (§3), each behaving like one "word."

---

## 5. The two matrices (and what each dimension means)

| Matrix | Shape (this sample) | What it is | Dim meanings |
|---|---|---|---|
| **Hidden states** `X` | `(4540, 2048)` | the token vectors flowing **between** layers | rows = sequence positions; cols = 2048 feature channels |
| **Attention scores** | `(4540, 4540)` per head | computed **inside** each layer from `X` | both axes = sequence positions; entry (i,j) = how much token i attends to token j |

The `(4540, 2048)` is **not** the attention matrix — it's the input *from which*
the `(4540, 4540)` is derived, each attention layer:

```
X : (4540, 2048)
Q = X·W_Q → 16 heads × (4540, 128)
K,V = X·W_K, X·W_V   # GQA: only 2 KV heads, each shared by 8 query heads
per head:  scores = Q·Kᵀ/√128 → (4540, 4540) → softmax → ·V → (4540, 128)
concat 16 heads → (4540, 2048) → out-proj → next layer
```

A `4540 × 4540` score matrix is formed **per head (16) per layer (36)**.

---

## 6. Causal attention

All layers are `full_attention` (no sliding window) but this is a **decoder LM**,
so the `4540 × 4540` is **causal / lower-triangular**: position *p* attends only to
positions ≤ *p*. **Token order is what realizes the fusion**: the 1305
`<|bev_token_*|>` sit *after* all 2990 image tokens and the prompt, so every bev
query can attend back to all image patches + the route/speed text, while the image
tokens cannot see the (later) bev queries. This is the paper's "deep self-attention"
fusion of vision + world queries + text, implemented purely by sequence position.

---

## 7. What the bev / pixel tokens become at the OUTPUT

(Useful for understanding their *purpose*.) After the 36 layers the output hidden
states split into two heads
([modeling_qwen2_5_vl.py ~1529–1543](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1529)):
- **`vis_head`** (`Linear 2048 → 1024`): applied at the `<|bev_token_*|>` positions →
  the predicted future-BEV latent, trained by MSE against frozen **DINOv3** features
  of the real future BEV image (the "world loss"). This is why bev tokens can be
  constant placeholders: their *output* carries the prediction.
- **`lm_head`** (shared, tied to the input embeddings): produces all text — including
  the `<|pixel_token_N|>` and numeric **waypoints** in the answer — by ordinary
  next-token prediction.

So: bev tokens → world-latent head; pixel/text tokens → language head.

---

## 8. Other insights worth knowing

- **Quadratic cost.** Attention is O(seq²): `4540² ≈ 20.6M` scores × 16 heads × 36
  layers. Materialized in fp32 that is ~1.3 GB *per layer*, which is why the run
  uses `sdpa`/FlashAttention (compute the weighted output without storing the full
  square). The sequence is dominated by the **2990 image + 1305 bev** tokens, so
  those drive the compute; fewer/smaller images quadratically reduce cost.
- **GQA.** 16 query heads but only 2 key/value heads (groups of 8 share K,V) — saves
  KV-cache memory; the score matrix is still `4540×4540` per query head.
- **mRoPE (multimodal RoPE).** `rope_scaling.mrope_section = [16,24,24]` — image
  tokens get 3-D (temporal, height, width) positions instead of a single 1-D index,
  so the model knows each patch's 2-D location within its frame.
- **Prefill vs decode.** The full `4540×4540` is the one-time **prefill** of the
  prompt. When generating the answer, the **KV-cache** means each new token computes
  only a `1 × current_len` row, not a fresh square — so decoding is cheap vs prefill.
- **The 5 BEV "target" images are never read at inference.** The JSONL lists 15
  images, but the prompt has only 10 `<image>` markers; `images[10:15]`
  (placeholders here) are training-only DINOv3 targets, not loaded for inference.

---

## 9. Quick-reference numbers (this sample)

| Quantity | Value |
|---|---|
| total sequence length | 4540 tokens |
| image tokens | 2990 (299 × 10 images) |
| bev tokens | 1305 (5 × 261) |
| pixel tokens (in input) | 12 (6 route points × 2) |
| text tokens (incl. chat template) | 211 |
| hidden size (per-token vector) | 2048 |
| pixel_values | (11960, 1176) = (10·26·46, 3·2·14·14) |
| image_grid_thw (per image) | [1, 26, 46] → 299 tokens |
| layers × heads | 36 × 16 (+2 KV heads) |
| head_dim | 128 |
| attention score matrix | 4540 × 4540 per head, causal |
| key token ids | image_pad 151655; bev 151671–152975; pixel 152976–153486 |

---

## 10. Q&A

### Q — What is a "target pixel token" and where do they come from? Why these values?
**They are the navigation route / goal — not the future trajectory.** They tell the
model *where to go* (like a sparse GPS line), and the model then plans the waypoints.

Source and construction (`bench2drive/dataprocess/targetpointgen.py`):
1. CARLA provides route **command points** in each frame's anno: `x/y_command_near`
   (the next route target) plus a final `x/y_command_far`. Verified for frame 20:
   `near = (2459.35, 2534.89)`, `far = (2459.72, 2574.89)` (world coords — note world-x
   ≈ constant, y increasing → a straight route).
2. Once **per scene**, all frames' `command_near` points (+ the final `far`) are
   collected → deduplicated → a turn segment is cropped → **equidistantly resampled
   (~10 m)** into a sparse route polyline.
3. **Per frame**, each route point (world coords) is projected into the *current* ego's
   **TOP-DOWN / BEV grid** with the same `world2cam` + quantization as §1 →
   `(dy, dx)` pixel-token pair; points outside `±255` are dropped.

So the **same world route is re-expressed in each frame's moving-ego BEV**. As the ego
drives forward, route points it has already passed move *behind* it (negative `dy`).
That explains the example values for frame 20:
```
[(-20,0),(35,0),(90,0),(145,0),(200,0),(254,0)]
```
- every `dx = 0` → the road is **straight ahead** (no lateral offset);
- `dy` spans **−20 (≈20 cells behind, already passed) → 254 (far ahead)**;
- spacing ≈ 55 cells between points ≈ the 10 m equidistant resampling.

Purpose: **goal conditioning**. Without it the model would know the scene but not which
way the route wants the car to go (straight vs. which exit). It's the discrete,
BEV-grounded counterpart of a navigation command.

### Q — Why are historical trajectories floats (not pixel tokens)? Why continuous?
The historical trajectory is the **past ego positions in metric meters**, expressed in
the *current* ego frame: `world2ego @ past_world_location → (x, y)` in metres, written
as text floats with 2 decimals (`parse_anno`). Example `(-7.68,0.00)` = 7.68 m **behind**,
0 lateral. Convention: **x = forward (signed), y = lateral**, so history is negative-x
(behind) and the predicted future waypoints are positive-x (ahead).

The design deliberately uses **two coordinate systems**:
- **Metric metres (ego frame)** for the historical trajectory *and* the output
  **future waypoints** — the physically meaningful driving units the PID controller
  actually consumes, and the units L2 is measured in.
- **BEV pixel cells** for the target / future *pixel* tokens — an image-grounded
  discretization tied to the BEV scene and the world-model latents.

Why floats rather than pixel tokens here:
1. **Past↔future consistency** — history and the predicted waypoints live in the same
   metric frame, so the model reasons about speed/dynamics in one physical system
   (it can literally see "−3.97 m → 0 → +3.79 m" implies ~8 m/s, matching the speed line).
2. **Precision** — control needs fine resolution (e.g. 3.79 m); the `±255` grid at
   ~2 px/cell is coarse and tied to the image.
3. The LLM already represents numbers natively as **digit tokens** (`"3"`,`"."`,`"7"`,
   `"9"`), so a float is just ordinary text — no special vocabulary needed.

"Continuous vs discrete": at the *token* level a float is still discrete (digit
characters), but it **represents a continuous metric value** at arbitrary precision,
whereas a pixel token can only land on one of 511 grid cells. So: floats = fine /
metric / continuous-valued; pixel tokens = coarse / BEV-image / quantized.

Note the **answer emits both**: `future pixel tokens` (discrete, BEV) *and*
`future waypoints` (continuous, metres) — a dual encoding (FSDrive-style). The
**waypoints in metres** are what's used downstream (L2 eval + PID control); the pixel
tokens ground the same path in the BEV-image / world-model space.

### Q — Is the training loss `loss_rec` the same as the eval **L2**? What does `loss_rec` actually average over?
**No — different quantities on different scales** (not two implementations of one thing).

**First, the forward step — how `logits` arise (hidden → head).** The 36 causal layers output
`hidden (1, L, 2048)`; then `hidden → final RMSNorm → lm_head (2048→153536)` produces
`logits (1, L, 153536)` — a next-token distribution at *every* position (the text/“language” head). In
parallel, the `<|bev_token|>` rows' 2048-vectors go `→ vis_head (2048→1024)` (the world head, §7). One hidden
matrix, two projection heads. **Training runs this in ONE parallel pass, not a token-by-token loop**: the
causal mask alone enforces the autoregressive conditioning (`logits[i]` sees only tokens `0..i`, so it predicts
`x_{i+1}`), and because the whole GT answer is available, all `L` next-token CEs are computed at once.
**Teacher forcing** = during training the model is fed the *ground-truth* previous tokens (never its own
predictions), which is exactly what makes that single parallel pass possible; its cost is the train/inference
“exposure bias” gap (at test time the model must instead consume its own outputs — see the inference Q below).

| | `loss_rec` (training) | **L2** (open-loop eval) |
|---|---|---|
| what | token **cross-entropy** (next-token) | **Euclidean distance** between predicted & GT waypoints |
| mode | **teacher-forced**, all positions in ONE parallel pass (GT fed in) | **autoregressive generation**, then decode + regex-parse |
| operates on | `lm_head` logits `(L, 153536)` → `−log p(correct next token)` | parsed `(x,y)` waypoint floats |
| where | `self.loss_function(logits, labels, …)` ([modeling_qwen2_5_vl.py:1540](src/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1540)) | `cal_l2_loss`/`parse_answer` ([eval_and_visual_local.py](src/tools/eval_and_visual_local.py)), via [eval_l2.py](src/tools/eval_l2.py) |
| unit | nats / token | metres |

**What `loss_rec` averages over — NOT all ~4540.** Logits exist at every position, but the CE is **masked**
(`labels = -100`) everywhere except the assistant **answer text** tokens. The 1305 `<|bev_token|>` are masked
too — they are supervised by the world loss (`loss_gen`/`vis_head`, §7), not the text CE. So for a no-CoT
sample `loss_rec` averages over only **~75 tokens**: `<think>None.</think>` (~4) + the pixel-token block
(~23: template + the **8** `<|pixel_token_N|>`) + the waypoint block (~48: template + the **digit tokens** of
the 4 `(x,y)`); with a real CoT, ~125–225. It is **dominated by easy template tokens**, so a small `loss_rec`
change maps only loosely to trajectory error — which is why the *policy* is judged by **L2 in metres**, not by
`loss_rec`. (Two L2 caveats: the eval "1s/2s" are ADE *up to* that horizon, not point-at-time, so our ~1.0 is
not directly comparable to the paper's 0.58; and any λ-ablation uses the same calculator on both arms, so the
*relative* comparison is robust regardless.)

**Why labels are shifted by one.** At position `i` the model has *already been given* `x_i` (the causal mask
includes `i`), so "predicting `x_i`" is trivial copying — it must predict the **unseen next** token `x_{i+1}`.
Aligning `logits[i] ↔ x_{i+1}` is the one-position shift (`logits[:, :-1]` vs `labels[:, 1:]`).

**What is back-propagated:** `loss = loss_rec + world_loss_weight · loss_gen`. **The L2 is NOT in the loss**
(eval-only, non-differentiable through token sampling + regex parse). The trajectory is learned **purely as
text**, via the CE on the waypoint digit tokens inside `loss_rec` — there is no separate trajectory/L2 head.

**Ground truth = the `role:"assistant"` message:** the text (CoT + pixel tokens + waypoint digits) is the
`loss_rec` target; the 1305 bev tokens' GT is DINOv3 features of the future-BEV images (`loss_gen`). The
`role:"user"` turn is **context, fully masked** (`-100`). ⚠ Don't confuse the prompt's *target* pixel tokens
(an input goal) with the assistant's *future* pixel tokens (the GT to produce).

### Q — How is the `(4540, 2048)` hidden state used "sequentially" at inference? And do test files' GT get used by the model?
**Training is parallel** (the whole GT answer is fed → teacher forcing → all next-token CEs in one pass).
**Inference has no GT**, so it is sequential — and the key point: **only the *last* row of the hidden state
predicts the next token** (recall `logits[i]` predicts token `i+1`). The `(4540, 2048)` is *not* "used
sequentially"; the sequence grows one row at a time and each step projects only the newest row:

1. **Prefill — one parallel pass** over prompt + 10 images + 1305 bev (= the **4540** input rows):
   `input_ids (1,4540)` → `inputs_embeds (1,4540,2048)` (embed + scatter, §3) → 36 causal layers →
   `hidden (1,4540,2048)`; **cache K/V for all 4540.** Use only `hidden[:, -1, :] (1,2048)` → `lm_head` →
   `logits (1,153536)` → sample the **first** answer token. (Rows 0..4538 aren't projected for generation —
   their logits would just re-predict prompt tokens; they only serve as cached attention context.)
2. **Decode loop — sequential**, until `</answer>`/EOS (or `max_new_tokens`): feed the 1 new token `(1,1)` →
   `embed (1,1,2048)` → causal layers **using the KV cache** (the new token attends to all cached past) →
   one new row `hidden (1,1,2048)` → `lm_head` → `logits (1,153536)` → next token → append → repeat.

So generation repeats **~75 times** (the *answer* length), **not** 4540: the ~4540 prompt/bev rows are
prefilled once; only the answer is generated. The final full sequence (~4540 + ~75 ≈ 4615) matches training's
length, but the autoregressive part is just the answer. The **KV cache** is what makes each decode step cost
~one token instead of re-forming the `4540×4540` square (§8). ("Concatenate → projector", by shape:
*concatenate* = build the fused `inputs_embeds (1,L,2048)` (§3/§4); *projector* = `lm_head : (·,2048)→(·,153536)`
— training applies it to all L rows at once, inference to one row per step.)

**Do test files contain the GT? Yes — but the model never sees it; only the scorer does.** `infer_local.py`
builds the model input from `messages[0]` (user) **only** — the assistant turn is commented out
([infer_local.py:71](src/infer_local.py#L71)) — generates from the prompt, and stores `messages[1]` separately
as `gt` ([:89](src/infer_local.py#L89)) so `eval_l2` can compare `pred` vs `gt`. The GT rides along with the
sample purely as the answer key for the L2 metric.

| split | file | GT **fed to the model**? | GT used for |
|---|---|---|---|
| **train** | `train.jsonl` | **yes** — teacher-forced (input *and* target) | back-propagated CE (`loss_rec`) + world MSE (`loss_gen`) |
| **eval** (during training) | `eval.jsonl` | **yes** — teacher-forced forward | `eval_loss` (CE), no backprop → early-stopping signal |
| **test** (final L2) | `test.jsonl` | **no** — generated from the prompt only | only the **scorer** (parse GT waypoints → L2 vs `pred`) |
