import collections
import dataclasses
import logging
import math
import pathlib
import json


# 修改 examples/libero/main.py
import sys
import os
# 添加 Conda 环境的 site-packages 目录
conda_env_path = os.path.join(os.path.expanduser('~'), 'anaconda3', 'envs', 'libero', 'lib', 'python3.8', 'site-packages')
sys.path.append(conda_env_path)

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10 #50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    # 添加恢复相关参数
    resume: bool = False  # 是否从上次中断处恢复
    resume_task_id: int = 0  # 恢复的任务 ID
    resume_episode_id: int = 0  # 恢复的 episode ID
    results_file: str = ""  # 结果文件名，如果为空，将根据任务套件名自动生成


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # 设置结果文件名
    if not args.results_file:
        args.results_file = f"results_{args.task_suite_name}.json"
    
    # 初始化结果字典
    results_path = pathlib.Path(args.video_out_path) / args.results_file
    if args.resume and results_path.exists():
        # 如果恢复模式且结果文件存在，加载之前的结果
        with open(results_path, "r") as f:
            results = json.load(f)
        total_episodes = results.get("total_episodes", 0)
        total_successes = results.get("total_successes", 0)
        logging.info(f"Resuming from task {args.resume_task_id}, episode {args.resume_episode_id}")
        logging.info(f"Loaded previous results: {total_successes}/{total_episodes} successes")
    else:
        # 否则初始化新的结果字典
        results = {
            "task_results": [],
            "total_episodes": 0,
            "total_successes": 0,
            "total_success_rate": 0.0
        }
        total_episodes, total_successes = 0, 0

    # 开始评估
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # 如果是恢复模式且当前任务 ID 小于恢复任务 ID，跳过
        if args.resume and task_id < args.resume_task_id:
            logging.info(f"Skipping task {task_id}")
            continue
            
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        
        # 查找或初始化当前任务的结果
        task_result = next((r for r in results["task_results"] if r["task_id"] == task_id), None)
        if task_result is None:
            task_result = {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
                "completed_episodes": []  # 记录已完成的 episode
            }
            results["task_results"].append(task_result)
        
        task_episodes = task_result["episodes"]
        task_successes = task_result["successes"]
        completed_episodes = task_result.get("completed_episodes", [])
        
        # 开始 episodes
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # 如果是恢复模式且当前任务是恢复任务且当前 episode 小于恢复 episode，跳过
            if args.resume and task_id == args.resume_task_id and episode_idx < args.resume_episode_id:
                logging.info(f"Skipping episode {episode_idx}")
                continue
                
            # 如果这个 episode 已经完成，跳过
            if episode_idx in completed_episodes:
                logging.info(f"Episode {episode_idx} already completed, skipping")
                continue
                
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {episode_idx+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # 记录这个 episode 已完成
            completed_episodes.append(episode_idx)
            task_result["completed_episodes"] = completed_episodes
            
            # 更新任务结果
            task_episodes += 1
            task_result["episodes"] = task_episodes
            if done:
                task_successes += 1
                task_result["successes"] = task_successes
            total_episodes += 1
            
            # 确保成功次数不超过尝试次数
            if task_successes > task_episodes:
                logging.warning(f"Success count ({task_successes}) exceeds episode count ({task_episodes}). Correcting.")
                task_successes = task_episodes
                task_result["successes"] = task_successes

            # 更新成功率
            task_result["success_rate"] = float(task_successes) / float(task_episodes)
            
            # 更新总结果
            results["total_episodes"] = total_episodes
            results["total_successes"] = total_successes
            results["total_success_rate"] = float(total_successes) / float(total_episodes)
            
            # 每完成一个 episode 就保存结果，以便中断后恢复
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

    # Add total results to results dictionary
    results["total_episodes"] = total_episodes
    results["total_successes"] = total_successes
    results["total_success_rate"] = float(total_successes) / float(total_episodes)

    # Save results to JSON file
    with open(f"{args.video_out_path}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
