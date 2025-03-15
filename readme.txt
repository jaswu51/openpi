huggingface-cli download --repo-type dataset --resume-download openvla/modified_libero_rlds --local-dir openvla/modified_libero_rlds
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir openvla/modified_libero_rlds --push_to_hub
