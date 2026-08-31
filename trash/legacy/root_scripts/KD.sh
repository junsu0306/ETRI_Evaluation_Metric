export CUDA_VISIBLE_DEVICES=0

#python main_for_KD.py --mode=cluster --arch=mobilenetv2 --pooling=netvlad --num_clusters=64


#python main_for_KD.py --mode=train --arch=mobilenetv2 --pooling=netvlad --num_clusters=64 --teacher_arch=resnet-18 --patience=10 

python main_for_KD.py --mode=test --arch=mobilenetv2 --resume=runs/Aug21_15-48-05_mobilenetv2_netvlad --split=val --ckpt=latest