CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-5/primiturbo_trd_group-5-4/knn-4__few_van_van+VAE__2dgs+nerf_opac-dpt_32-0.02+32-0.1_D__512+4096_p-0_s-0_lr_1e-4__TRD-PM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"