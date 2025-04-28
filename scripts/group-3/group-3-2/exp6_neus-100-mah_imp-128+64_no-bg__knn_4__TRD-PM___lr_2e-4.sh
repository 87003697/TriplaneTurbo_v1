CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-3/primiturbo_trd_group-3-2/neus-100-mah_imp-128+64_no-bg__knn_4__TRD-PM___lr_2e-4.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"