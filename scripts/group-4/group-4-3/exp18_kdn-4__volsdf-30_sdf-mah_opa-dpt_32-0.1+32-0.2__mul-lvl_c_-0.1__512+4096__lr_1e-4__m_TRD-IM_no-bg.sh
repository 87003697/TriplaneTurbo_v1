CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-4/primiturbo_trd_group-4-3/kdn-4__volsdf-30_sdf-mah_opa-dpt_32-0.1+32-0.2__mul-lvl_c_-0.1__512+4096__lr_1e-4__m_TRD-IM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"