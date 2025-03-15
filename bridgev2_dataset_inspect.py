import tensorflow as tf
import os
import json
import glob
from tqdm import tqdm

def extract_language_instructions(directory_path, output_file):
    """
    遍历目录中的所有 TFRecord 文件，提取语言指令，并按 episode ID 保存到 JSON 文件
    """
    print(f"正在从目录中提取语言指令: {directory_path}")
    
    # 获取目录中的所有 TFRecord 文件
    tfrecord_files = glob.glob(os.path.join(directory_path, "*.tfrecord-*"))
    print(f"找到 {len(tfrecord_files)} 个 TFRecord 文件")
    
    # 用于存储结果的字典
    results = {
        "files": {},
        "total_episodes": 0,
        "unique_instructions": [],
        "episodes": {}
    }
    
    # 遍历所有文件
    global_episode_id = 0
    for file_path in tqdm(tfrecord_files, desc="处理文件"):
        file_name = os.path.basename(file_path)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            continue
        
        # 创建 TFRecord 数据集
        dataset = tf.data.TFRecordDataset(file_path)
        
        # 提取该文件中的所有语言指令
        file_episodes = {}
        episode_count = 0
        
        for serialized_example in dataset:
            example = tf.train.Example()
            example.ParseFromString(serialized_example.numpy())
            feature_dict = example.features.feature
            
            episode_id = f"{file_name}_{episode_count}"
            global_episode_id += 1
            episode_count += 1
            
            # 尝试提取语言指令
            instruction = ""
            
            # 首先检查常见的语言指令字段
            instruction_keys = ['steps/language_instruction', 'language_instruction', 'instruction']
            
            for key in instruction_keys:
                if key in feature_dict:
                    feature = feature_dict[key]
                    if feature.WhichOneof('kind') == 'bytes_list' and feature.bytes_list.value:
                        try:
                            instruction = feature.bytes_list.value[0].decode('utf-8')
                            break
                        except UnicodeDecodeError:
                            pass
            
            # 保存该 episode 的指令
            file_episodes[episode_id] = instruction
            results["episodes"][f"episode_{global_episode_id}"] = {
                "file": file_name,
                "local_id": episode_count - 1,
                "instruction": instruction
            }
            
            # 如果找到了新的指令，添加到唯一指令列表中
            if instruction and instruction not in results["unique_instructions"]:
                results["unique_instructions"].append(instruction)
        
        # 将该文件的结果添加到总结果中
        results["files"][file_name] = {
            "episode_count": episode_count,
            "episodes": file_episodes
        }
        
        results["total_episodes"] += episode_count
    
    # 对唯一指令列表进行排序
    results["unique_instructions"].sort()
    results["unique_instruction_count"] = len(results["unique_instructions"])
    
    # 保存结果到 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成。共处理了 {results['total_episodes']} 个示例，找到 {results['unique_instruction_count']} 个不同的语言指令。")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    # 指定目录路径和输出文件
    directory_path = "/home/wuyi/Documents/github/openpi/bridge_data_v2/bridge_dataset/bridge_dataset/1.0.0"
    output_file = "bridge_language_instructions_by_episode.json"
    
    extract_language_instructions(directory_path, output_file)