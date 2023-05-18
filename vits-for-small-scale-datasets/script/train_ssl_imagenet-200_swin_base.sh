cd /work/workspace12/chenhuan/code/vits-for-small-scale-datasets-main
source activate mmdet


output_dir=${root_path}/outputs/ssl/swin_base_imagenet-200
python -m torch.distributed.launch --master_port 61510 train_ssl.py --arch swin \
                                   --dataset Tiny_Imagenet --image_size 64 \
                                   --datapath "/path/to/tiny-imagenet/train/folder" \
                                   --patch_size 4  \
                                   --mlp_head_in 384 \
                                   --local_crops_number 8 \
                                   --local_crops_scale 0.2 0.4 \
                                   --global_crops_scale 0.5 1. 
                                   --out_dim 1024 \
                                   --batch_size_per_gpu 256  \
                                   --output_dir "/path/for/saving/checkpoints"

# baseline +trans
export CUDA_VISIBLE_DEVICES=0
root_path=/work/workspace12/chenhuan/code
datapath=${root_path}/datasets/tiny-imagenet-200
pretrained_weights=${root_path}/outputs/ssl/swin_base_imagenet-200/checkpoint.pth
output_dir=${root_path}/outputs/ssl/swin_base_imagenet-200
python finetune.py --arch swin  \
                   --dataset Tiny-Imagenet \
                   --datapath ${datapath} \
                   --batch_size 128 \
                   --epochs 300 \
                   --pretrained_weights ${pretrained_weights} \
                   --output_dir ${output_dir} \
                   --model swin_base  --tag vfsd_baseline

export CUDA_VISIBLE_DEVICES=0
root_path=/work/workspace12/chenhuan/code
datapath=${root_path}/datasets/tiny-imagenet-200
pretrained_weights=${root_path}/outputs/ssl/swin_base_imagenet-200/checkpoint.pth
output_dir=${root_path}/outputs/ssl/swin_base_imagenet-200
python finetune.py --arch swin  \
                   --dataset Tiny-Imagenet \
                   --datapath ${datapath} \
                   --batch_size 256 \
                   --epochs 300 \
                   --output_dir ${output_dir} \
                   --model swin_base  --tag baseline