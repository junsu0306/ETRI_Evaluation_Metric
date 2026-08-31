export CUDA_VISIBLE_DEVICES=3




python main_for_teacher.py --mode=cluster --arch=resnet-18 --pooling=netvlad --num_clusters=64
