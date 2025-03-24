"""
可视化Bridge Data V2 TFRecord文件中的图像

使用方法:
python visualize_bridge_images.py --file_path /path/to/tfrecord --step 0
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm

def decode_image(image_string):
    """将图像字符串解码为numpy数组"""
    image = tf.image.decode_jpeg(image_string, channels=3)
    return image.numpy()

def visualize_step(file_path, step_index=0, save_dir=None):
    """可视化TFRecord文件中指定步骤的图像"""
    print(f"正在从文件中读取图像: {file_path}")
    print(f"可视化步骤: {step_index}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    # 创建TFRecord数据集
    dataset = tf.data.TFRecordDataset(file_path)
    
    # 获取第一个示例
    for serialized_example in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        feature_dict = example.features.feature
        
        # 获取总步骤数
        total_steps = len(feature_dict['steps/observation/image_0'].bytes_list.value)
        print(f"总步骤数: {total_steps}")
        
        # 检查步骤索引是否有效
        if step_index >= total_steps:
            print(f"错误: 步骤索引 {step_index} 超出范围 (0-{total_steps-1})")
            return
        
        # 获取语言指令（如果有）
        instruction = None
        if 'steps/language_instruction' in feature_dict:
            try:
                instruction = feature_dict['steps/language_instruction'].bytes_list.value[step_index].decode('utf-8')
                print(f"语言指令: {instruction}")
            except:
                print("无法解码语言指令")
        
        # 创建图像网格
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 解码并显示四个相机视角的图像
        for i, cam_idx in enumerate(range(4)):
            key = f'steps/observation/image_{cam_idx}'
            if key in feature_dict:
                image_string = feature_dict[key].bytes_list.value[step_index]
                image = decode_image(image_string)
                
                row, col = i // 2, i % 2
                axes[row, col].imshow(image)
                axes[row, col].set_title(f'Camera {cam_idx}')
                axes[row, col].axis('off')
        
        # 获取动作
        action = None
        if 'steps/action' in feature_dict:
            # 动作是一个扁平化的数组，每个步骤有7个值
            action_values = feature_dict['steps/action'].float_list.value
            action_dim = 7  # 假设每个动作是7维的
            start_idx = step_index * action_dim
            end_idx = start_idx + action_dim
            if end_idx <= len(action_values):
                action = action_values[start_idx:end_idx]
        
        # 设置标题
        if instruction:
            plt.suptitle(f'指令: {instruction}\n步骤: {step_index+1}/{total_steps}', fontsize=14)
        else:
            plt.suptitle(f'步骤: {step_index+1}/{total_steps}', fontsize=14)
        
        # 显示动作信息
        if action:
            action_text = f'动作: {np.array2string(np.array(action), precision=4)}'
            plt.figtext(0.5, 0.01, action_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        # 保存图像
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, f'step_{step_index}.png')
            plt.savefig(output_path)
            print(f'图像已保存到: {output_path}')
        
        # 显示图像
        plt.show()
        break

def visualize_sequence(file_path, start_step=0, num_steps=5, save_dir=None):
    """可视化TFRecord文件中的一系列步骤"""
    print(f"正在从文件中读取图像序列: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    # 创建保存目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 创建TFRecord数据集
    dataset = tf.data.TFRecordDataset(file_path)
    
    # 获取第一个示例
    for serialized_example in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        feature_dict = example.features.feature
        
        # 获取总步骤数
        total_steps = len(feature_dict['steps/observation/image_0'].bytes_list.value)
        print(f"总步骤数: {total_steps}")
        
        # 调整步骤范围
        end_step = min(start_step + num_steps, total_steps)
        
        # 获取语言指令（如果有）
        instruction = None
        if 'steps/language_instruction' in feature_dict:
            try:
                instruction = feature_dict['steps/language_instruction'].bytes_list.value[0].decode('utf-8')
                print(f"语言指令: {instruction}")
            except:
                print("无法解码语言指令")
        
        # 可视化每个步骤
        for step_idx in tqdm(range(start_step, end_step), desc="处理步骤"):
            # 创建图像网格
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 解码并显示四个相机视角的图像
            for i, cam_idx in enumerate(range(4)):
                key = f'steps/observation/image_{cam_idx}'
                if key in feature_dict:
                    image_string = feature_dict[key].bytes_list.value[step_idx]
                    image = decode_image(image_string)
                    
                    row, col = i // 2, i % 2
                    axes[row, col].imshow(image)
                    axes[row, col].set_title(f'Camera {cam_idx}')
                    axes[row, col].axis('off')
            
            # 获取动作
            action = None
            if 'steps/action' in feature_dict:
                # 动作是一个扁平化的数组，每个步骤有7个值
                action_values = feature_dict['steps/action'].float_list.value
                action_dim = 7  # 假设每个动作是7维的
                start_idx = step_idx * action_dim
                end_idx = start_idx + action_dim
                if end_idx <= len(action_values):
                    action = action_values[start_idx:end_idx]
            
            # 设置标题
            if instruction:
                plt.suptitle(f'指令: {instruction}\n步骤: {step_idx+1}/{total_steps}', fontsize=14)
            else:
                plt.suptitle(f'步骤: {step_idx+1}/{total_steps}', fontsize=14)
            
            # 显示动作信息
            if action:
                action_text = f'动作: {np.array2string(np.array(action), precision=4)}'
                plt.figtext(0.5, 0.01, action_text, ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, bottom=0.1)
            
            # 保存图像
            if save_dir:
                output_path = os.path.join(save_dir, f'step_{step_idx}.png')
                plt.savefig(output_path)
                print(f'步骤 {step_idx+1} 图像已保存到: {output_path}')
            
            plt.close()
        
        print(f"序列可视化完成，共处理了 {end_step - start_step} 个步骤")
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化Bridge Data V2 TFRecord文件中的图像')
    parser.add_argument('--file_path', type=str, required=True, help='TFRecord文件路径')
    parser.add_argument('--step', type=int, default=0, help='要可视化的步骤索引（默认为0，即第一步）')
    parser.add_argument('--sequence', action='store_true', help='是否可视化一系列步骤')
    parser.add_argument('--start_step', type=int, default=0, help='序列可视化的起始步骤')
    parser.add_argument('--num_steps', type=int, default=5, help='要可视化的步骤数量')
    parser.add_argument('--save_dir', type=str, default='visualization_output', help='保存图像的目录')
    
    args = parser.parse_args()
    
    if args.sequence:
        visualize_sequence(args.file_path, args.start_step, args.num_steps, args.save_dir)
    else:
        visualize_step(args.file_path, args.step, args.save_dir)