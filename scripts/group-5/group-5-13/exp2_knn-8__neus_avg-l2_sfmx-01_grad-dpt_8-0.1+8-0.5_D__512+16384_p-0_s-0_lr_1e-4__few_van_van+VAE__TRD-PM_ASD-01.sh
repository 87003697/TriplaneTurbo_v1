CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-5/primiturbo_trd_group-5-13/knn-8__neus_avg-l2_sfmx-01_grad-dpt_8-0.1+8-0.5_D__512+16384_p-0_s-0_lr_1e-4__few_van_van+VAE__TRD-PM_ASD-01.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"