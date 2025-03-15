import tensorflow as tf
import os
import json
import glob
from tqdm import tqdm

def extract_wipe_sweep_episodes(input_directory, output_directory, instructions_json):
    """
    提取包含 'wipe' 或 'sweep' 关键词的示例，并保存到指定文件夹
    如果文件已存在，则跳过保存，只更新索引
    """
    print(f"正在从目录中提取包含 'wipe' 或 'sweep' 的示例: {input_directory}")
    
    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)
    
    # 加载语言指令 JSON 文件
    with open(instructions_json, 'r', encoding='utf-8') as f:
        instructions_data = json.load(f)
    
    # 找出包含关键词的示例
    target_episodes = {}
    keywords = ['wipe', 'sweep']
    
    # 创建索引数据结构
    index = {
        "metadata": {
            "keywords": keywords,
            "total_episodes": 0,
            "source_directory": input_directory,
            "output_directory": output_directory,
            "skipped_files": 0,
            "new_files": 0
        },
        "episodes": {}
    }
    
    for file_name, file_info in instructions_data['files'].items():
        for episode_id, instruction in file_info['episodes'].items():
            if any(keyword in instruction.lower() for keyword in keywords):
                if file_name not in target_episodes:
                    target_episodes[file_name] = []
                
                # 提取本地 episode ID（数字部分）
                local_id = int(episode_id.split('_')[-1])
                target_episodes[file_name].append({
                    'episode_id': episode_id,
                    'local_id': local_id,
                    'instruction': instruction
                })
                
                # 添加到索引
                output_file_name = f"{episode_id}.tfrecord"
                index["episodes"][episode_id] = {
                    "file": file_name,
                    "local_id": local_id,
                    "instruction": instruction,
                    "output_file": output_file_name,
                    "keywords": [keyword for keyword in keywords if keyword in instruction.lower()]
                }
    
    # 统计找到的示例数量
    total_episodes = sum(len(episodes) for episodes in target_episodes.values())
    index["metadata"]["total_episodes"] = total_episodes
    print(f"找到 {total_episodes} 个包含关键词的示例")
    
    # 提取并保存目标示例
    for file_name, episodes in tqdm(target_episodes.items(), desc="处理文件"):
        input_file_path = os.path.join(input_directory, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(input_file_path):
            print(f"错误: 文件 {input_file_path} 不存在")
            continue
        
        # 创建 TFRecord 数据集
        dataset = None  # 延迟加载数据集，只在需要时加载
        examples = None  # 延迟加载示例，只在需要时加载
        
        # 提取并保存每个目标示例
        for episode in episodes:
            local_id = episode['local_id']
            episode_id = episode['episode_id']
            instruction = episode['instruction']
            
            # 创建输出文件名
            output_file_name = f"{episode_id}.tfrecord"
            output_file_path = os.path.join(output_directory, output_file_name)
            
            # 检查文件是否已存在
            if os.path.exists(output_file_path):
                print(f"跳过已存在的文件: {output_file_name}")
                index["metadata"]["skipped_files"] += 1
                continue
            
            # 延迟加载数据集和示例
            if dataset is None:
                dataset = tf.data.TFRecordDataset(input_file_path)
                examples = list(dataset)
            
            # 确保索引在范围内
            if local_id >= len(examples):
                print(f"警告: 索引 {local_id} 超出文件 {file_name} 的范围")
                continue
            
            # 获取示例
            example = examples[local_id]
            
            # 保存示例到单独的文件
            with tf.io.TFRecordWriter(output_file_path) as writer:
                writer.write(example.numpy())
            
            print(f"已保存: {output_file_name} - 指令: {instruction}")
            index["metadata"]["new_files"] += 1
    
    # 保存索引 JSON 文件
    index_file_path = instructions_wipe_sweep_json
    with open(index_file_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成。已提取 {total_episodes} 个示例到 {output_directory}")
    print(f"跳过了 {index['metadata']['skipped_files']} 个已存在的文件")
    print(f"新保存了 {index['metadata']['new_files']} 个文件")
    print(f"索引文件已保存到: {index_file_path}")

if __name__ == "__main__":
    # 指定目录路径和输出文件
    input_directory = "/home/wuyi/Documents/github/openpi/bridge_data_v2/bridge_dataset/bridge_dataset/1.0.0"
    output_directory = "/home/wuyi/Documents/github/openpi/bridge_data_v2/bridge_dataset/bridge_dataset_preprocessed"
    instructions_json = "bridge_language_instructions_by_episode.json"
    instructions_wipe_sweep_json = "/home/wuyi/Documents/github/openpi/bridge_language_instructions_by_episode_wipe_sweep.json"
    
    extract_wipe_sweep_episodes(input_directory, output_directory, instructions_json)