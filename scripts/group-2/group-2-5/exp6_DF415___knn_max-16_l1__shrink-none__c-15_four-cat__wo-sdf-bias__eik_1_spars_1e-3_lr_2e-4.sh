CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-2/primiturbo_trd_group-2-5/DF415___knn_max-16_l1__shrink-none__c-15_four-cat__wo-sdf-bias__eik_1_spars_1e-3_lr_2e-4.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"