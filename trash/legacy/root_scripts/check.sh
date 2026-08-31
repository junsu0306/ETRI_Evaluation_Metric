export CUDA_VISIBLE_DEVICES=2


python check_parameter.py --mode=test --resume=runs/May01_14-48-56_resnet-18_netvlad --split=val --ckpt=best

