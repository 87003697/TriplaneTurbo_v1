CUDA_VISIBLE_DEVICES=2 python launch.py \
    --config configs/gaussianturbo_prd_group-1/DF415_debug_v1.yaml  \
    --train \
    data.prompt_library="dreamfusion_415_prompt_library"

