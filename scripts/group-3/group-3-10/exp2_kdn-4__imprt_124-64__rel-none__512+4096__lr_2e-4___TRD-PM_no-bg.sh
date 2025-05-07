# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
#     --config configs/primiturbo_trd_group-3/primiturbo_trd_group-3-10/kdn-4__imprt_124-64__rel-none__512+4096__lr_2e-4___TRD-PM_no-bg.yaml \
#     --train \
#   data.prompt_library="dreamfusion_415_prompt_library"

CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/primiturbo_trd_group-3/primiturbo_trd_group-3-10/kdn-4__imprt_124-64__rel-none__512+4096__lr_2e-4___TRD-PM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"