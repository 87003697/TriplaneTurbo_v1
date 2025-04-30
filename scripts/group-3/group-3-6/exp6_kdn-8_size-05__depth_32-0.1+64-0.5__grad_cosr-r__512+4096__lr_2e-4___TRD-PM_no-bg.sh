CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-3/primiturbo_trd_group-3-6/kdn-8_size-05__depth_32-0.1+64-0.5__grad_cosr-r__512+4096__lr_2e-4___TRD-PM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"