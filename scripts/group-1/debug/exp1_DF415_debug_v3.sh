# CUDA_VISIBLE_DEVICES=0 python launch.py \
#     --config configs/primiturbo_trd_group-1/debug/DF415_debug_v3.yaml  \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"
# 0,1,3,4,

# TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5,6 python launch.py \
#     --config configs/primiturbo_trd_group-1/debug/DF415_debug_v3.yaml  \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"


# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
#     --config configs/primiturbo_trd_group-1/debug/DF415_debug_v3.yaml  \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
#     --config configs/primiturbo_trd_group-1/debug/DF415_debug_v3.yaml  \
#     --train \
#     data.prompt_library="dreamfusion_415_prompt_library"

CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/primiturbo_trd_group-1/debug/DF415_debug_v3.yaml  \
    --train \
    data.prompt_library="dreamfusion_415_prompt_library"
