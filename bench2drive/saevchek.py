import os
import torch
import json
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration,AutoProcessor

def save_qwen25vl_from_gpu(model, tokenizer,save_dir="./qwen25vl_gpu_backup2", ref_config_path="/mnt/nas-data-1/zhanglingjun.zlj1/modelversion/v3_bev_target_fulldata_resume/checkpoint-19813"):
    """
    从 GPU 直接保存 Qwen2.5VL 模型权重到本地（无需原始文件路径）
    
    参数:
        model: 已加载到 GPU 的 Qwen2.5VL 模型对象
        save_dir: 保存目录路径
        ref_config_path: 配置参考路径（用于获取完整配置）
    """
    # 1. 验证模型位置
    current_device = next(model.parameters()).device
    print(f"✅ 模型当前在 {current_device} 上")
    
    # 2. 确保参考配置路径有效
    if not os.path.exists(ref_config_path):
        raise FileNotFoundError(f"参考配置路径不存在: {ref_config_path}")
    print(f"🔍 使用参考配置: {ref_config_path}")

    # 3. 检查内存并安全转移权重到 CPU
    print("\n🔍 检查 CPU 内存...")
    try:
        # 尝试全精度 (FP32)
        print("尝试全精度 (FP32) 保存...")
        model = model.cpu()
        print("✅ 全精度转移成功 (需 ~12GB RAM)")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠️ 内存不足！改用半精度 (FP16) 保存 (需 ~6GB RAM)")
            model = model.half().cpu()
        else:
            raise

    # 4. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n📦 创建保存目录: {save_dir}")

    # 5. 复制参考配置文件（关键！因为原始文件已删除）
    print(" COPYING REFERENCE CONFIG FILES...")
    config_files = [
        "config.json", 
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.model"
    ]
    
    for file in config_files:
        src = os.path.join(ref_config_path, file)
        dst = os.path.join(save_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (跳过)")

    # 6. 保存模型权重（处理大文件）
    print("\nSAVING MODEL WEIGHTS...")
    state_dict = model.state_dict()
    
    # 使用 Hugging Face 官方分片方法
    # from transformers.modeling_utils import shard_checkpoint
    state_dict = model.state_dict()
    
    # ========== 修复点：替换分片逻辑 ==========
    print("⚠️ 使用兼容分片方案 (支持旧版 Transformers)...")
    max_shard_bytes = 5 * 1024**3  # 2GB
    current_shard = {}
    current_size = 0
    sharded_state = {}
    index = {"weight_map": {}}

    for name, tensor in state_dict.items():
        size = tensor.numel() * tensor.element_size()
        if size > max_shard_bytes:
            shard_name = f"pytorch_model.{name.replace('.', '_')}.bin"
            sharded_state[shard_name] = {name: tensor}
            index["weight_map"][name] = shard_name
            continue
            
        if current_size + size > max_shard_bytes and current_shard:
            shard_name = f"pytorch_model-{len(sharded_state)+1:05d}-of-99999.bin"
            sharded_state[shard_name] = current_shard
            for k in current_shard.keys():
                index["weight_map"][k] = shard_name
            current_shard, current_size = {}, 0
        
        current_shard[name] = tensor
        current_size += size

    if current_shard:
        shard_name = f"pytorch_model-{len(sharded_state)+1:05d}-of-99999.bin"
        sharded_state[shard_name] = current_shard
        for k in current_shard.keys():
            index["weight_map"][k] = shard_name
    
    index["metadata"] = {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())}
    # ========== 修复结束 ==========
    
    
    # 保存每个分片
    for shard_file, shard in tqdm(sharded_state.items(), desc="Writing shards"):
        shard_path = os.path.join(save_dir, shard_file)
        torch.save(shard, shard_path)
    
    # 保存索引文件
    if index:
        index_path = os.path.join(save_dir, "pytorch_model.bin.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"✨ 生成索引文件: {index_path}")

    # 7. 保存 Qwen2.5VL 专用组件（确保多模态支持）
    print("\nSAVING MULTIMODAL COMPONENTS...")
    multimodal_files = []
    
    if hasattr(model, 'visual_encoder'):
        vis_state = model.visual_encoder.state_dict()
        vis_path = os.path.join(save_dir, "visual_encoder.bin")
        torch.save(vis_state, vis_path)
        multimodal_files.append("visual_encoder.bin")
    
    if hasattr(model, 'mm_projector'):
        proj_state = model.mm_projector.state_dict()
        proj_path = os.path.join(save_dir, "mm_projector.bin")
        torch.save(proj_state, proj_path)
        multimodal_files.append("mm_projector.bin")
    
    print("  ✓ " + ", ".join(multimodal_files) if multimodal_files else "  ✓ No multimodal components found")

    # 8. 验证保存结果
    print("\n✅ 保存完成！验证文件列表:")
    total_size = 0
    for root, _, files in os.walk(save_dir):
        for file in files:
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            total_size += size
            print(f"  - {os.path.relpath(path, save_dir)} | {size/(1024**2):.2f} MB")
    
    print(f"\n📦 总大小: {total_size/(1024**3):.2f} GB")
    print(f"🎉 模型已安全保存到: {os.path.abspath(save_dir)}")
    print("💡 后续加载命令:")
    print(f"   model = Qwen2_5VLForConditionalGeneration.from_pretrained('{save_dir}', device_map='auto')")
    return save_dir

# ======================
# 你的模型加载代码（无需修改）
# ======================
if __name__ == "__main__":
    # 这是你的原始加载代码（已确认模型在 GPU 7 上）
    model_path = "/mnt/nas-data-1/zhanglingjun.zlj1/modelversion/v3_bev_target_fulldata_resume/checkpoint-19000"
    
    # 注意：虽然路径存在，但权重已被覆盖，我们需要从当前GPU提取
    print("=" * 60)
    print("Qwen2.5VL GPU 7 模型备份工具 (原始文件已删除)")
    print("=" * 60)
    print(f"⚠️ 注意: 模型路径 {model_path} 的权重已被覆盖，正在从GPU提取...")
    
    # 加载模型（你的原始代码）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"\n✅ 模型已加载到 {next(model.parameters()).device}")
    print(f"模型类型: {model.__class__.__name__}")
    
    # ======================
    # 从 GPU 提取并保存
    # ======================
    save_dir = save_qwen25vl_from_gpu_bf16(
        model,
        tokenizer,  # 必须传入 tokenizer
        save_dir="./qwen25vl_gpu7_bf16",
        ref_config_path=model_path
    )
    
    # ======================
    # 保存 processor（关键！）
    # ======================
    processor.save_pretrained(save_dir)
    print(f"\n✅ Processor 已保存到: {save_dir}")
    
    # ======================
    # 验证保存（确保可加载）
    # ======================
    print("\n🔍 验证: 尝试从本地加载模型...")
    try:
        loaded_model = Qwen2_5VLForConditionalGeneration.from_pretrained(
            save_dir,
            device_map="auto",
            torch_dtype=torch.float16 if model.dtype == torch.float16 else torch.float32
        )
        print(f"✅ 加载成功！设备: {next(loaded_model.parameters()).device}")
        
        # 检查多模态组件
        if hasattr(loaded_model, 'visual_encoder'):
            print("  ✓ 视觉编码器已恢复")
        if hasattr(loaded_model, 'mm_projector'):
            print("  ✓ 多模态投影层已恢复")
            
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        print("可能原因: 1. 内存不足 2. 配置不完整 3. 依赖版本问题")
        exit(1)
    
    print("\n" + "="*60)
    print("✨ 备份完成！请使用以下命令加载模型:")
    print(f"   from transformers import Qwen2_5VLForConditionalGeneration")
    print(f"   model = Qwen2_5VLForConditionalGeneration.from_pretrained('{save_dir}', device_map='auto')")
    print("="*60)
