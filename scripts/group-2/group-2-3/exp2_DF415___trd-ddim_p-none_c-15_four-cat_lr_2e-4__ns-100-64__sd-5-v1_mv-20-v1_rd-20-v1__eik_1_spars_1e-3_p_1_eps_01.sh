CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-2/primiturbo_trd_group-2-3/DF415___trd-ddim_p-none_c-15_four-cat_lr_2e-4__ns-100-64__sd-5-v1_mv-20-v1_rd-20-v1__eik_1_spars_1e-3_p_1_eps_01.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"