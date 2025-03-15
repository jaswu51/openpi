"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro
import os
from huggingface_hub import HfApi, snapshot_download
import time

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    # 设置更长的超时时间和更多的重试次数
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 增加到300秒
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_MAX_RETRIES"] = "5"
    
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # 添加断点续传功能
    local_dir = os.path.join(os.getenv("HF_HOME", "~/.cache/huggingface"), "datasets", data_config.repo_id)
    os.makedirs(local_dir, exist_ok=True)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            dataset = _data_loader.create_dataset(data_config, config.model)
            dataset = _data_loader.TransformedDataset(
                dataset,
                [
                    *data_config.repack_transforms.inputs,
                    *data_config.data_transforms.inputs,
                    # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
                    RemoveStrings(),
                ],
            )
            return data_config, dataset
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed, waiting 30 seconds before retrying...")
            time.sleep(30)  # 增加等待时间到30秒


def main(config_name: str, max_frames: int | None = None):
    try:
        config = _config.get_config(config_name)
        data_config, dataset = create_dataset(config)

        num_frames = len(dataset)
        shuffle = False

        if max_frames is not None and max_frames < num_frames:
            num_frames = max_frames
            shuffle = True

        data_loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=8,#1,
            num_workers=8,
            shuffle=shuffle,
            num_batches=num_frames,
        )

        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_frames, desc="Computing stats"):
            for key in keys:
                values = np.asarray(batch[key][0])
                stats[key].update(values.reshape(-1, values.shape[-1]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

        output_path = config.assets_dirs / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Try these solutions:")
        print("1. Check your internet connection")
        print("2. Use a VPN or proxy if needed")
        print("3. Try running with:")
        print("   export HTTPS_PROXY=your_proxy_here")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        raise


if __name__ == "__main__":
    tyro.cli(main)
