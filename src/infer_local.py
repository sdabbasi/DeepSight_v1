import os
import json
import argparse
import cv2
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


# 初始化模型
def init_model(model_path, attn_implementation="sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    model.eval()
    sampling_params = None
    return sampling_params, processor, model, tokenizer


def format_message(sample):
    content = sample['messages'][0]['content']
    content = content.split('<image>')
    assert len(content) == 11
    format_content = []
    for i in range(len(content)):
        c = {
            "type": "text",
            "text": content[i]
        }
        format_content.append(c)
        if i != len(content) - 1:
            c = {
                "type": "image",
                "image": sample["images"][i],
                "resized_height":364, "resized_width":644
            }
            format_content.append(c)
    return format_content


def add_bev_text(text):
    t, h, w, patchsize, n_cls, n_register = 5, 256, 256, 16, 1, 4
    l = t * (h * w // (patchsize ** 2) + n_cls + n_register)
    bev_content = []
    for i in range(l):
        bev_content.append(f"<|bev_token_{i}|>")
    bev_content = ''.join(bev_content)

    # Prefill ONLY the BEV block (no <think> prefix), matching the released
    # checkpoint's training format (BEV-first, then <think>) and the closed-loop
    # agent's add_bev_text (bench2drive/team_code/qwen_b2d_agent.py). The model
    # then autoregressively emits <think>…</think> + the two <answer> blocks.
    text = text + '<|start_bev_token|>' + bev_content + '<|end_bev_token|>\n'

    return text

# 推理一个patch
def infer_one_patch(tokenizer, processor, model, val_sample, max_new_tokens=512):
    # 构造message：
    messages = [
        {
            "role": "user",
            "content": format_message(val_sample)
        },
        # {
        #     "role": "assistant",
        #     "content": '<think> None </think>\n<|start_bev_token|>'
        # }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 他的作用是？
    text = add_bev_text(text)
    # from pudb import set_trace; set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text, images=[image_inputs], videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    # Output
    output_text = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    output_data_one = {
        'prompt': val_sample['messages'][0]['content'],
        'gt': val_sample['messages'][1]['content'].split('<|end_bev_token|>\n')[1],
        'pred': output_text
    }
    # carry through scene/frame tags (if the builder added them) for per-scene eval
    for k in ('_scene', '_frame'):
        if k in val_sample:
            output_data_one[k] = val_sample[k]
    return output_data_one


if __name__ == '__main__':
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Offline (open-loop) DeepSight inference over a local sharegpt JSONL.")
    parser.add_argument('--model_path', default=os.path.join(repo, 'checkpoints', 'deepsight'),
                        help='Path to the (patched Qwen2.5-VL + DINOv3) checkpoint.')
    parser.add_argument('--val_data', default=os.path.join(repo, 'local_data', 'infer_samples.jsonl'),
                        help='sharegpt JSONL with messages + 10 input images per sample.')
    parser.add_argument('--out', default=os.path.join(repo, 'debug', 'infer_results.json'),
                        help='Output JSONL: one {prompt, gt, pred} object per line.')
    parser.add_argument('--limit', type=int, default=0, help='Cap number of samples (0 = all).')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Decode budget for CoT + waypoints AFTER the prefilled BEV block.')
    parser.add_argument('--attn', default='sdpa', choices=['sdpa', 'eager', 'flash_attention_2'])
    parser.add_argument('--index', type=int, default=0, help='Shard index for multi-process sharding.')
    parser.add_argument('--num_pro', type=int, default=1, help='Number of shards.')
    args = parser.parse_args()

    val_datas = [json.loads(line) for line in open(args.val_data) if line.strip()]
    if args.num_pro > 1:
        n_per_proc = len(val_datas) // args.num_pro + 1
        val_datas = val_datas[args.index * n_per_proc: (args.index + 1) * n_per_proc]
    if args.limit:
        val_datas = val_datas[:args.limit]
    print(f'samples to infer: {len(val_datas)} (shard {args.index}/{args.num_pro}, limit={args.limit})')

    print(f'loading model from {args.model_path} (attn={args.attn}) ...')
    sampling_params, processor, model, tokenizer = init_model(args.model_path, attn_implementation=args.attn)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results = []
    for val_sample in tqdm(val_datas):
        result = infer_one_patch(tokenizer, processor, model, val_sample, max_new_tokens=args.max_new_tokens)
        results.append(json.dumps(result, ensure_ascii=False))
        with open(args.out, 'w') as fp:   # incremental flush so a crash keeps partial results
            fp.write('\n'.join(results))
    print(f'wrote {len(results)} results -> {args.out}')
