#!/usr/bin/env python
"""Build a sharegpt JSONL for *offline / open-loop* DeepSight inference from
locally-extracted official Bench2Drive scene folders.

Why this exists
---------------
The released ModelScope dataset (`hotdogcheese/b2d_deepsight`) is a 65 GB
text-only JSONL whose `images` are dead NAS paths — it has no actual pixels.
The official Bench2Drive scenes (`rethinklab/Bench2Drive`, one `.tar.gz` per
scene) DO contain the camera images. This script turns such a local scene into
the exact sharegpt format the released checkpoint expects, reusing the
authoritative prompt/answer/projection logic from
`bench2drive/dataprocess/targetpointgen.py` (the script that produced the
released training data).

Format notes (verified against the released data + the closed-loop agent):
  * prompt encodes the navigation goal as "target pixel tokens" (route points
    projected into BEV pixels) + historical trajectory + speed + a
    `<CoT_flag_*>` toggle. There is NO "Mission Goal" string.
  * assistant is BEV-first: `<|start_bev_token|>…<|end_bev_token|>` then
    `<think>…</think>` then the two `<answer>` blocks. We only need it for the
    ground-truth future waypoints used by the open-loop L2 eval.
  * we have no Qwen3-VL CoT annotations locally, so FLAGE='False'
    (`<CoT_flag_False>` / `<think>None.</think>`).

Inference only consumes the 10 input images (4 history `rgb_front` + 6 surround);
the 5 future-BEV images are training-only DINOv3 targets, so we fill those slots
with a placeholder image that is never read.

Usage:
    python src/tools/build_local_infer_jsonl.py \
        --scenes local_data/bench2drive_raw/AccidentTwoWays_Town12_Route1102_Weather10 \
        --out local_data/infer_samples.jsonl --stride 20 --limit 20
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import the authoritative builder functions (its __main__ guard prevents the NAS run)
sys.path.insert(0, os.path.join(REPO, "bench2drive", "dataprocess"))
import targetpointgen as tpg  # noqa: E402

PLACEHOLDER_IMG = os.path.join(REPO, "data", "empty_img.jpg")
SURROUND = ["front", "front_left", "front_right", "back", "back_left", "back_right"]


def _frame_num(path: str) -> int:
    return int(re.search(r"(\d+)", os.path.basename(path)).group(1))


def load_scene_annos(scene: str):
    """Return annos as a list indexed by frame number (0..N-1, contiguous)."""
    files = sorted(glob.glob(os.path.join(scene, "anno", "*.json.gz")), key=_frame_num)
    opener = gzip.open
    if not files:  # fall back to plain .json
        files = sorted(glob.glob(os.path.join(scene, "anno", "*.json")), key=_frame_num)
        opener = open
    if not files:
        raise FileNotFoundError(f"no anno/*.json[.gz] under {scene}")
    nums = [_frame_num(f) for f in files]
    assert nums == list(range(nums[0], nums[-1] + 1)), f"non-contiguous frames in {scene}"
    annos = []
    for f in files:
        with opener(f, "rt") as fh:
            annos.append(json.load(fh))
    return annos


def get_images_infer(i: int, scene: str):
    """10 input images (4 history rgb_front + 6 surround) + 5 BEV placeholders."""
    history_path = os.path.join(scene, "camera", "rgb_front")
    images = []
    for k in range(4, 0, -1):  # 2.0s,1.5s,1.0s,0.5s ago
        his = i - k * 5
        images.append(os.path.join(history_path, f"{his:05d}.jpg") if his >= 0 else PLACEHOLDER_IMG)
    for cam in SURROUND:
        images.append(os.path.join(scene, "camera", f"rgb_{cam}", f"{i:05d}.jpg"))
    images += [PLACEHOLDER_IMG] * 5  # future-BEV target slots, never loaded at inference
    assert len(images) == 15
    return images


def route_sampled_points(annos):
    """Reproduce targetpointgen.create_train_json's route-point sampling."""
    nums = len(annos)
    target_points = []
    for i in range(1, nums - 20):
        p = (annos[i]["x_command_near"], annos[i]["y_command_near"])
        if p not in target_points:
            target_points.append(p)
    if nums - 20 >= 0 and nums - 20 < nums:
        far = (annos[nums - 20]["x_command_far"], annos[nums - 20]["y_command_far"])
        if far not in target_points:
            target_points.append(far)
    reduced, _ = tpg.extract_straight_lanes(target_points, angle_threshold=0.05)
    sampled = tpg.plot_sampled_points(reduced, angle_threshold=15, max_extension=25, label="route")
    if not sampled:  # degenerate route -> fall back to raw target points
        sampled = target_points if target_points else [(0.0, 0.0)]
    return sampled


def build_scene(scene: str, stride: int, limit: int, start: int):
    annos = load_scene_annos(scene)
    nums = len(annos)
    sampled = route_sampled_points(annos)
    out = []
    # start at >=20 so all 4 history frames exist (no placeholders among inputs)
    lo = max(start, 20)
    for i in range(lo, nums - 20, stride):
        try:
            (his, speed, command, _think, bev, fpix, ftraj,
             _xt, _yt, _zt, bevtp) = tpg.parse_anno(i, annos, sampled)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {os.path.basename(scene)} frame {i}: {e}")
            continue
        prompt = tpg.get_prompt(command=command, his_trajs=his, speed_content=speed,
                                bevtargetpoints=bevtp, FLAGE="False")
        answer = tpg.get_answer(bev_content=bev, future_trajs_pixel=fpix,
                                future_trajs=ftraj, FLAGE="False")
        out.append({
            "messages": [
                {"content": prompt, "role": "user"},
                {"content": answer, "role": "assistant"},
            ],
            "images": get_images_infer(i, scene),
            "_scene": os.path.basename(scene),
            "_frame": i,
        })
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="One or more extracted scene folders (containing anno/ and camera/).")
    ap.add_argument("--out", default=os.path.join(REPO, "local_data", "infer_samples.jsonl"))
    ap.add_argument("--stride", type=int, default=20, help="Sample every Nth frame within a scene.")
    ap.add_argument("--limit", type=int, default=20, help="Max samples per scene (0 = all).")
    ap.add_argument("--start", type=int, default=20, help="First frame index to consider.")
    args = ap.parse_args()

    assert os.path.exists(PLACEHOLDER_IMG), f"missing placeholder image {PLACEHOLDER_IMG}"
    all_samples = []
    for scene in args.scenes:
        scene = scene.rstrip("/")
        print(f"building from {scene} ...")
        s = build_scene(scene, args.stride, args.limit, args.start)
        print(f"  -> {len(s)} samples")
        all_samples.extend(s)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
