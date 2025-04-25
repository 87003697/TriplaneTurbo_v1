CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-3/primiturbo_trd_group-3-1/DF415___attr-tex_knn_max-08_l1__shrink-none_no-bg__c-15_four-add__basic-bias__eik_01_lr_2e-4.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"