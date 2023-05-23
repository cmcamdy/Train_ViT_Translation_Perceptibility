cd /work/workspace12/chenhuan/code/SPT_LSA_ViT-main
source activate mmdet

# tiny-imagenet-200 spt lsa
export CUDA_VISIBLE_DEVICES=4
data_path=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
dataset=T-IMNET
batch_size=128
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa 
python main.py --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag}

tag=vit-spt-lsa-open
python main.py --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

tag=vit-spt-lsa-open-swin
python main.py --model swin --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT


# imagenet-100 spt lsa
export CUDA_VISIBLE_DEVICES=1
dataset='T-IMNET-100'
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/imagenet-100
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img100
python main.py --model vit --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

export CUDA_VISIBLE_DEVICES=2
dataset=T-IMNET-100
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/imagenet-100
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img100-swin
python main.py --model swin --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

# imagenet-100 spt lsa trans
export CUDA_VISIBLE_DEVICES=1
dataset=T-IMNET-100
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/imagenet-100
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img100-affine_trans-vit
python main_affine.py --with_trans --model vit --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

export CUDA_VISIBLE_DEVICES=2
dataset=T-IMNET-100
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/imagenet-100
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img100-swin-affine_trans-swin
python main_affine.py --with_trans --model swin --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

# imagenet-100 spt lsa scale
export CUDA_VISIBLE_DEVICES=3
dataset=T-IMNET-100
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/imagenet-100
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img100-affine_scale-vit
python main_affine.py --with_scale --model vit --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

export CUDA_VISIBLE_DEVICES=4
dataset=T-IMNET-100
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/imagenet-100
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img100-swin-affine_scale-swin
python main_affine.py --with_scale --model swin --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT


# tiny-imagenet-200 spt lsa trans
export CUDA_VISIBLE_DEVICES=0
dataset='T-IMNET'
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img200-affine_trans
python main_affine.py --with_trans --model vit --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

export CUDA_VISIBLE_DEVICES=2
dataset='T-IMNET'
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img200-affine_trans-swin
python main_affine.py  --with_trans --model swin --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT



# tiny-imagenet-200 spt lsa scale
export CUDA_VISIBLE_DEVICES=0
dataset='T-IMNET'
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img200-affine_scale
python main_affine.py --with_scale --model vit --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT

export CUDA_VISIBLE_DEVICES=1
dataset='T-IMNET'
batch_size=128
data_path=/work/workspace12/chenhuan/code/datasets/tiny-imagenet-200
save_path=/work/workspace12/chenhuan/code/outputs/spt_lsa
tag=vit-spt-lsa-open-img200-affine_scale-swin
python main_affine.py  --with_scale --model swin --data_path ${data_path} --dataset ${dataset} --batch_size ${batch_size} --save_path ${save_path} --tag ${tag} --is_LSA --is_SPT