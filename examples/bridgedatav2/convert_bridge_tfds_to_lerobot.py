"""
将Bridge Data V2数据集转换为LeRobot格式的脚本。

数据是在WidowX 250 6DOF机器人臂上收集的。

使用方法:
uv run examples/bridgedatav2/convert_bridge_data_v2_to_lerobot.py --data_dir /path/to/your/data

如果您想将数据集推送到Hugging Face Hub，可以使用以下命令:
uv run examples/bridgedatav2/convert_bridge_data_v2_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

注意: 运行此脚本需要安装tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`
"""

import shutil
import os
from pathlib import Path

# 修复导入问题
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# 直接定义HF_LEROBOT_HOME路径
HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"
import tensorflow_datasets as tfds
import tyro
import numpy as np

REPO_NAME = "jaswu51/bridge_data_v2"  # 输出数据集的名称，也用于Hugging Face Hub
RAW_DATASET_NAMES = [
    "bridge_data_v2",
]  # Bridge Data V2数据集名称

def main(data_dir: str, *, push_to_hub: bool = False):
    # 清理输出目录中的任何现有数据集
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # 创建LeRobot数据集，定义要存储的特征
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="widowx250",  # WidowX 250机器人
        fps=5,
        features={
            "image_0": {
                "dtype": "image",
                "shape": (256, 256, 3),  # 根据features.json的图像尺寸
                "names": ["height", "width", "channel"],
            },
            "image_1": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "image_2": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "image_3": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),  # 根据features.json的状态维度
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # 根据features.json的动作维度
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 加载Bridge Data V2数据集
    for raw_dataset_name in RAW_DATASET_NAMES:
        # 从指定目录加载数据集构建器
        builder = tfds.builder_from_directory(builder_dir=data_dir)
        raw_dataset = builder.as_dataset(split="train")
        
        # 计数处理的episodes
        episode_count = 0
        
        for episode in raw_dataset:
            episode_count += 1
            if episode_count % 100 == 0:
                print(f"已处理 {episode_count} 个episodes...")
            
            # 获取语言指令（假设每个episode有相同的指令）
            first_step = next(iter(episode["steps"]))
            language_instruction = ""
            if "natural_language_instruction" in first_step["observation"]:
                language_instruction = first_step["observation"]["natural_language_instruction"].numpy().decode()
            elif "language_instruction" in first_step:
                language_instruction = first_step["language_instruction"].numpy().decode()
            
            # 从语言指令中提取任务名称，如果没有则使用默认值
            task_name = f"task_{episode_count}"  # 默认任务名称
            
            for step in episode["steps"].as_numpy_iterator():
                # 处理观察和动作
                observation = step["observation"]
                action = step["action"]
                
                # 提取状态 - 假设状态包含关节位置
                state = observation["state"]
                
                # 处理动作 - 根据WidowX 250的控制方式调整
                # 假设动作包含关节位置目标和夹爪命令
                if isinstance(action, dict):
                    # 如果动作是字典格式
                    if "world_vector" in action and "rotation_delta" in action:
                        # 如果使用笛卡尔空间控制
                        world_vector = action["world_vector"].numpy()
                        rotation_delta = action["rotation_delta"].numpy()
                        gripper = float(action["open_gripper"].numpy()) if "open_gripper" in action else 0.0
                        
                        # 合并为7维动作向量
                        combined_action = np.concatenate([
                            world_vector,          # 位置变化 (3维)
                            rotation_delta,        # 旋转变化 (3维)
                            np.array([gripper]),   # 夹爪状态 (1维)
                        ])
                    elif "joint_positions" in action:
                        # 如果使用关节空间控制
                        joint_positions = action["joint_positions"].numpy()
                        gripper = float(action["gripper_position"].numpy()) if "gripper_position" in action else 0.0
                        
                        # 合并为7维动作向量
                        combined_action = np.concatenate([
                            joint_positions,       # 关节位置 (6维)
                            np.array([gripper]),   # 夹爪状态 (1维)
                        ])
                else:
                    # 如果动作是直接的数组
                    combined_action = action
                
                # 提取图像 - 处理多个相机视角
                images = {}
                for i in range(4):  # 假设有4个相机视角
                    key = f"image_{i}"
                    if key in observation:
                        images[key] = observation[key]
                
                # 构建帧数据字典，不包含task
                frame_data = {
                    "state": state,
                    "actions": combined_action,
                }
                
                # 添加所有可用的图像
                for key, img in images.items():
                    frame_data[key] = img
                
                dataset.add_frame(frame_data)
            
            # 保存episode时传递task参数
            current_task = str(language_instruction) if language_instruction else task_name
            try:
                dataset.save_episode(task=current_task)
            except ValueError as e:
                print(f"保存episode时出错: {e}")
                # 尝试不带task保存
                dataset.clear_episode_buffer()
                # 重新处理episode的帧，但不包含task
                for step in episode["steps"].as_numpy_iterator():
                    observation = step["observation"]
                    action = step["action"]
                    state = observation["state"]
                    
                    # 处理动作 (与之前相同)
                    if isinstance(action, dict):
                        if "world_vector" in action and "rotation_delta" in action:
                            world_vector = action["world_vector"].numpy()
                            rotation_delta = action["rotation_delta"].numpy()
                            gripper = float(action["open_gripper"].numpy()) if "open_gripper" in action else 0.0
                            combined_action = np.concatenate([
                                world_vector,
                                rotation_delta,
                                np.array([gripper]),
                            ])
                        elif "joint_positions" in action:
                            joint_positions = action["joint_positions"].numpy()
                            gripper = float(action["gripper_position"].numpy()) if "gripper_position" in action else 0.0
                            combined_action = np.concatenate([
                                joint_positions,
                                np.array([gripper]),
                            ])
                    else:
                        combined_action = action
                    
                    # 提取图像
                    images = {}
                    for i in range(4):
                        key = f"image_{i}"
                        if key in observation:
                            images[key] = observation[key]
                    
                    # 构建不包含task的帧数据
                    frame_data = {
                        "state": state,
                        "actions": combined_action,
                    }
                    
                    # 添加所有可用的图像
                    for key, img in images.items():
                        frame_data[key] = img
                    
                    dataset.add_frame(frame_data)
                
                # 再次尝试保存episode，使用默认任务名称
                dataset.save_episode(task=task_name)
        
        print(f"总共处理了 {episode_count} 个episodes")

    # 整合数据集，与Libero脚本保持一致
    # dataset.consolidate(run_compute_stats=True)

    # # 可选择推送到Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["bridge_data_v2", "widowx250", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)