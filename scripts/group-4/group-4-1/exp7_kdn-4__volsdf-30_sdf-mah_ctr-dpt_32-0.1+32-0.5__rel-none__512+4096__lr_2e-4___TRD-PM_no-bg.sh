CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-4/primiturbo_trd_group-4-1/kdn-4__volsdf-30_sdf-mah_ctr-dpt_32-0.1+32-0.5__rel-none__512+4096__lr_2e-4___TRD-PM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"