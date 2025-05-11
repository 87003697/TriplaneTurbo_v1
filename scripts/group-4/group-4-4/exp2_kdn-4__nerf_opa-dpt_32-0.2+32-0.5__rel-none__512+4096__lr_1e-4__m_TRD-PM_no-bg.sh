CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python launch.py \
    --config configs/primiturbo_trd_group-4/primiturbo_trd_group-4-4/kdn-4__nerf_opa-dpt_32-0.2+32-0.5__rel-none__512+4096__lr_1e-4__m_TRD-PM_no-bg.yaml \
    --train \
  data.prompt_library="dreamfusion_415_prompt_library"