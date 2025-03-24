"""
检查Bridge Data V2数据集格式的脚本。

此脚本加载一个episode并打印其内容结构，帮助理解数据格式。

使用方法:
uv run examples/bridgedatav2/check_format.py --data_dir /path/to/your/data

注意: 运行此脚本需要安装tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`
"""

import os
from pathlib import Path
import json
import tensorflow_datasets as tfds
import tyro
import numpy as np
import matplotlib.pyplot as plt

def print_structure(obj, prefix="", is_last=True, max_depth=3, current_depth=0):
    """递归打印对象结构"""
    if current_depth > max_depth:
        return
    
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{type(obj).__name__}", end="")
    
    if isinstance(obj, dict):
        print(f" (keys: {len(obj)})")
        new_prefix = prefix + ("    " if is_last else "│   ")
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            print_structure(key, new_prefix, False, max_depth, current_depth + 1)
            print_structure(value, new_prefix, i == len(items) - 1, max_depth, current_depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f" (length: {len(obj)})")
        if len(obj) > 0:
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_structure(obj[0], new_prefix, True, max_depth, current_depth + 1)
    elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
        # 处理numpy数组或tensorflow张量
        print(f" (shape: {obj.shape}, dtype: {obj.dtype})")
    elif isinstance(obj, (int, float, str, bool)):
        print(f" (value: {obj})")
    else:
        print()

def visualize_images(step_data):
    """可视化一个step中的四个相机视角图像"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 获取四个图像
    images = []
    for i in range(4):
        key = f"image_{i}"
        if key in step_data["observation"]:
            img = step_data["observation"][key].numpy()
            images.append((key, img))
            
            # 检查image_3是否为深度图像
            if key == "image_3":
                # 检查图像的维度和特性
                print(f"\n检查 {key} 是否为深度图像:")
                print(f"形状: {img.shape}")
                print(f"数据类型: {img.dtype}")
                print(f"通道数: {1 if len(img.shape) == 2 else img.shape[2]}")
                print(f"最小值: {np.min(img)}")
                print(f"最大值: {np.max(img)}")
                print(f"平均值: {np.mean(img)}")
                
                # 深度图像通常是单通道的，或者值的分布与RGB图像不同
                is_depth = len(img.shape) == 2 or (img.shape[2] == 1)
                print(f"是否可能是深度图像: {'是' if is_depth else '否，可能是RGB图像'}")
    
    # 在2x2网格中显示图像
    for idx, (title, img) in enumerate(images):
        row, col = idx // 2, idx % 2
        
        # 对于可能的深度图像使用不同的显示方式
        if title == "image_3" and (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)):
            # 使用jet颜色映射显示深度图像
            if len(img.shape) == 3:
                img = img[:,:,0]  # 如果是3D但只有1个通道，转为2D
            im = axes[row, col].imshow(img, cmap='jet')
            fig.colorbar(im, ax=axes[row, col], label='深度值')
            axes[row, col].set_title(f"{title} - 深度图 ({img.shape})")
        else:
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{title} ({img.shape})")
        
        axes[row, col].axis('off')
    
    # 如果有语言指令，显示在图表标题中
    if "natural_language_instruction" in step_data["observation"]:
        instruction = step_data["observation"]["natural_language_instruction"].numpy()
        if hasattr(instruction, 'decode'):
            instruction = instruction.decode('utf-8')
        plt.suptitle(f"指令: {instruction}", fontsize=16)
    
    plt.tight_layout()
    # 保存到当前目录
    save_path = Path.cwd() / 'bridge_data_images.png'
    plt.savefig(save_path)
    print(f"图像已保存到当前目录: {save_path}")

def save_all_images(episode, sample_id):
    """保存episode中所有step的所有图像"""
    # 创建保存图像的目录，使用样本ID命名
    save_dir = Path.cwd() / f'bridge_data_images_sample_{sample_id}'
    save_dir.mkdir(exist_ok=True)
    print(f"\n创建图像保存目录: {save_dir}")
    
    # 遍历所有steps
    for step_idx, step in enumerate(episode["steps"]):
        # 遍历每个step中的所有图像
        for img_idx in range(4):  # 假设有4个相机视角
            img_key = f"image_{img_idx}"
            if img_key in step["observation"]:
                img = step["observation"][img_key].numpy()
                
                # 创建文件名
                filename = f"image_{img_idx}_step_{step_idx}.png"
                filepath = save_dir / filename
                
                # 保存图像
                plt.figure(figsize=(8, 8))
                
                # 对于可能的深度图像使用不同的显示方式
                if img_key == "image_3" and (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)):
                    if len(img.shape) == 3:
                        img = img[:,:,0]  # 如果是3D但只有1个通道，转为2D
                    plt.imshow(img, cmap='jet')
                    plt.colorbar(label='深度值')
                    plt.title(f"{img_key} - 深度图 (Step {step_idx})")
                else:
                    plt.imshow(img)
                    plt.title(f"{img_key} (Step {step_idx})")
                
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()
                
                print(f"保存图像: {filename}")

def main(data_dir: str):
    print(f"正在从 {data_dir} 加载数据集...")
    
    # 从指定目录加载数据集构建器
    builder = tfds.builder_from_directory(builder_dir=data_dir)
    
    # 获取数据集信息
    info = builder.info
    print(f"\n数据集信息:")
    print(f"名称: {info.name}")
    print(f"描述: {info.description}")
    print(f"版本: {info.version}")
    print(f"特征: {info.features}")
    print(f"训练集大小: {info.splits['train'].num_examples} 个样本")
    
    # 加载数据集
    raw_dataset = builder.as_dataset(split="train")
    
    # 处理第1000个样本
    sample_id = 999  # 索引从0开始，所以第1000个是索引999
    
    # 跳过前999个样本，获取第1000个
    target_sample = raw_dataset.skip(sample_id).take(1)
    
    for episode in target_sample:
        print(f"\n==== 样本 {sample_id+1} 结构 ====")
        print_structure(episode)
        
        # 获取第一个step
        first_step = next(iter(episode["steps"]))
        
        # 可视化第一个step的四个相机视角
        visualize_images(first_step)
        
        # 保存所有steps中的所有图像
        save_all_images(episode, sample_id+1)
        
        # 检查observation
        print("\n观察 (Observation):")
        for key, value in first_step["observation"].items():
            if hasattr(value, "numpy"):
                value_np = value.numpy()
                print(f"  {key}: 类型={type(value_np)}, 形状={value_np.shape if hasattr(value_np, 'shape') else '标量'}, 数据类型={value_np.dtype if hasattr(value_np, 'dtype') else type(value_np)}")
                
                # 对于图像，显示一些统计信息
                if key == "image" and hasattr(value_np, 'shape') and len(value_np.shape) == 3:
                    print(f"    图像统计: 最小值={value_np.min()}, 最大值={value_np.max()}, 平均值={value_np.mean():.2f}")
                
                # 对于文本数据，尝试解码
                if hasattr(value_np, 'dtype') and value_np.dtype.kind == 'S':
                    try:
                        decoded = value_np.decode('utf-8')
                        print(f"    解码内容: {decoded}")
                    except:
                        pass
            else:
                print(f"  {key}: {value}")
        
        # 检查action
        print("\n动作 (Action):")
        if isinstance(first_step["action"], dict):
            for key, value in first_step["action"].items():
                if hasattr(value, "numpy"):
                    value_np = value.numpy()
                    print(f"  {key}: 类型={type(value_np)}, 形状={value_np.shape if hasattr(value_np, 'shape') else '标量'}, 数据类型={value_np.dtype if hasattr(value_np, 'dtype') else type(value_np)}")
                else:
                    print(f"  {key}: {value}")
        else:
            action_np = first_step["action"].numpy()
            print(f"  动作向量: 形状={action_np.shape}, 数据类型={action_np.dtype}")
            print(f"  值: {action_np}")
        
        # 检查是否有语言指令
        if "natural_language_instruction" in first_step["observation"]:
            instruction = first_step["observation"]["natural_language_instruction"].numpy()
            if hasattr(instruction, 'decode'):
                instruction = instruction.decode('utf-8')
            print(f"\n语言指令: {instruction}")
        elif "language_instruction" in first_step:
            instruction = first_step["language_instruction"].numpy()
            if hasattr(instruction, 'decode'):
                instruction = instruction.decode('utf-8')
            print(f"\n语言指令: {instruction}")
        
        # 检查episode中的steps数量
        steps_count = sum(1 for _ in episode["steps"])
        print(f"\nEpisode中的steps数量: {steps_count}")
        
        # 打印episode中的所有键
        print("\nEpisode中的所有键:")
        for key in episode.keys():
            print(f"  - {key}")

if __name__ == "__main__":
    tyro.cli(main)