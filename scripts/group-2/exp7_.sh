CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-1/DF415_debug_v2.yaml  \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"