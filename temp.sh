export CUDA_VISIBLE_DEVICES=1


#python main_variant.py --mode=test --arch=mobilenetv2  --split=test --ckpt=best
python main_variant.py --mode=test --arch=mobilenetv2 --resume=runs/Aug18_07-43-17_mobilenetv2_netvlad --split=test --ckpt=best