CUDA_VISIBLE_DEVICES=0,1,2,3  python launch.py \
    --config custom/threestudio-3dgs/configs/gaussian_splatting_mvdream.yaml \
    --train \
    system.prompt_processor.prompt="an astronaut riding a horse"
