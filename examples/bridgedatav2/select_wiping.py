"""
从Bridge Data V2数据集中选择擦拭相关任务的脚本。

此脚本使用Claude API分析自然语言指令，判断是否为擦拭相关任务，
并将符合条件的样本ID保存到JSON文件中。

使用方法:
uv run examples/bridgedatav2/select_wiping.py --data_dir /path/to/your/data --api_key your_claude_api_key

注意: 运行此脚本需要安装tensorflow_datasets和anthropic:
`uv pip install tensorflow tensorflow_datasets anthropic`
"""

import os
from pathlib import Path
import json
import tensorflow_datasets as tfds
import tyro
import numpy as np
import time
from anthropic import Anthropic
from tqdm import tqdm

def is_wiping_task(instruction, client):
    """使用Claude API判断指令是否为擦拭相关任务"""
    try:
        prompt = f"""
        请判断以下机器人任务指令是否与擦拭(wiping)相关。
        擦拭相关任务包括：清洁表面、擦拭污渍、抹去灰尘、擦桌子、擦窗户等。
        
        指令: "{instruction}"
        
        只需回答"是"或"否"。
        """
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response.content[0].text.strip().lower()
        return "是" in answer or "yes" in answer
    except Exception as e:
        print(f"API调用出错: {e}")
        # 如果API调用失败，等待一段时间后重试
        time.sleep(5)
        return is_wiping_task(instruction, client)

def main(data_dir: str, api_key: str, output_file: str = "wiping_tasks.json", sample_limit: int = None):
    print(f"正在从 {data_dir} 加载数据集...")
    
    # 初始化Claude客户端
    client = Anthropic(api_key=api_key)
    
    # 从指定目录加载数据集构建器
    builder = tfds.builder_from_directory(builder_dir=data_dir)
    
    # 获取数据集信息
    info = builder.info
    print(f"\n数据集信息:")
    print(f"名称: {info.name}")
    print(f"描述: {info.description}")
    print(f"版本: {info.version}")
    print(f"训练集大小: {info.splits['train'].num_examples} 个样本")
    
    # 加载数据集
    raw_dataset = builder.as_dataset(split="train")
    
    # 存储擦拭相关任务的样本ID
    wiping_tasks = []
    
    # 确定要处理的样本数量
    total_samples = info.splits['train'].num_examples
    if sample_limit:
        total_samples = min(total_samples, sample_limit)
    
    print(f"\n开始分析 {total_samples} 个样本...")
    
    # 遍历数据集中的样本
    for sample_id, episode in tqdm(enumerate(raw_dataset.take(total_samples)), total=total_samples):
        # 获取第一个step
        first_step = next(iter(episode["steps"]))
        
        # 检查是否有语言指令
        instruction = None
        if "natural_language_instruction" in first_step["observation"]:
            instruction_raw = first_step["observation"]["natural_language_instruction"].numpy()
            if hasattr(instruction_raw, 'decode'):
                instruction = instruction_raw.decode('utf-8')
            else:
                instruction = str(instruction_raw)
        elif "language_instruction" in first_step:
            instruction_raw = first_step["language_instruction"].numpy()
            if hasattr(instruction_raw, 'decode'):
                instruction = instruction_raw.decode('utf-8')
            else:
                instruction = str(instruction_raw)
        
        if instruction:
            # 使用Claude API判断是否为擦拭任务
            if is_wiping_task(instruction, client):
                print(f"\n找到擦拭任务 (样本ID: {sample_id}):")
                print(f"指令: {instruction}")
                
                # 将样本ID和指令添加到结果列表
                wiping_tasks.append({
                    "sample_id": sample_id,
                    "instruction": instruction
                })
    
    # 保存结果到JSON文件
    output_path = Path.cwd() / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(wiping_tasks, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析完成！")
    print(f"共找到 {len(wiping_tasks)} 个擦拭相关任务")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    tyro.cli(main)