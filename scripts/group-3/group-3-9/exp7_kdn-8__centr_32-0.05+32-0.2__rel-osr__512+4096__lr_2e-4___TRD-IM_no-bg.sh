CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-3/primiturbo_trd_group-3-9/kdn-8__centr_32-0.05+32-0.2__rel-osr__512+4096__lr_2e-4___TRD-IM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"