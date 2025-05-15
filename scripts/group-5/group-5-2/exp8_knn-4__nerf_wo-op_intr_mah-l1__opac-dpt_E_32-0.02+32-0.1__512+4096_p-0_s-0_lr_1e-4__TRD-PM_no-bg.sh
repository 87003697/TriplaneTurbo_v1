CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-5/primiturbo_trd_group-5-2/knn-4__nerf_wo-op_intr_mah-l1__opac-dpt_E_32-0.02+32-0.1__512+4096_p-0_s-0_lr_1e-4__TRD-PM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"