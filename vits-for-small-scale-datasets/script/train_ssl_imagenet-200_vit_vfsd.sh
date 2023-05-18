cd /work/workspace12/chenhuan/code/vits-for-small-scale-datasets-office
source activate mmdet
root_path=/work/workspace12/chenhuan/code

# export CUDA_VISIBLE_DEVICES=0
# datapath=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
# output_dir=/work/workspace12/chenhuan/code/outputs/ssloff/vit_base_imagenet-200
# nohup python -m torch.distributed.launch --master_port 61411 train_ssl.py --arch vit \
#                                    --dataset Tiny-Imagenet --image_size 64 \
#                                    --datapath ${datapath} \
#                                    --patch_size 4  \
#                                    --mlp_head_in 192 \
#                                    --local_crops_number 8 \
#                                    --local_crops_scale 0.2 0.4 \
#                                    --global_crops_scale 0.5 1. \
#                                    --out_dim 1024 \
#                                    --batch_size_per_gpu 64  \
#                                    --output_dir ${output_dir} \
#                                    > ../logs/img200_vfsd_pretrain.log 2>&1 &


export CUDA_VISIBLE_DEVICES=5
datapath=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
pretrained_weights=${root_path}/outputs/ssloff/vit_base_imagenet-200/checkpoint0190.pth
output_dir=/work/workspace12/chenhuan/code/outputs/ssloff/vit_base_imagenet-200
nohup python finetune.py --arch vit  \
                   --dataset Tiny-Imagenet \
                   --datapath ${datapath} \
                   --batch_size 128 \
                   --epochs 300 \
                   --pretrained_weights ${pretrained_weights} \
                   --output_dir ${output_dir} \
                   --tag vfsd_baseline > \
                   ../logs/img200_vfsd_base_ssl_off.log 2>&1 &


export CUDA_VISIBLE_DEVICES=4
datapath=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
pretrained_weights=${root_path}/outputs/ssloff/vit_base_imagenet-200/checkpoint0190.pth
output_dir=/work/workspace12/chenhuan/code/outputs/ssloff/vit_base_imagenet-200
nohup python finetune_affine.py --arch vit  \
                   --dataset Tiny-Imagenet \
                   --datapath ${datapath} \
                   --batch_size 128 \
                   --epochs 300 \
                   --output_dir ${output_dir} \
                   --pretrained_weights ${pretrained_weights} \
                   --tag vfsd_with_trans \
                   --ls --lr 0.002 \
                   --alpha_trans 0.4 \
                   --init_weight 1.2 \
                   --with_trans \
                   > ../logs/img200_vfsd_with_trans_ssl_off.log 2>&1 &


export CUDA_VISIBLE_DEVICES=3
datapath=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
output_dir=/work/workspace12/chenhuan/code/outputs/ssloff/vit_base_imagenet-200
nohup python finetune_affine.py --arch vit  \
                   --dataset Tiny-Imagenet \
                   --datapath ${datapath} \
                   --batch_size 128 \
                   --epochs 300 \
                   --output_dir ${output_dir} \
                   --tag vfsd_with_scale_test \
                   --ls --lr 0.002 \
                   --alpha_trans 0.4 \
                   --init_weight 1.2 \
                   --with_scale \
                   > ../logs/img200_vfsd_with_scale_ssl_off.log 2>&1 &