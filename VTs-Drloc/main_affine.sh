export CUDA_VISIBLE_DEVICES=1
DATA_DIR=/work/workspace12/chenhuan/code/


IMG_SIZE=32 # 224, 384
MODE=vitcs # swintiny, cvt13, t2t, resnet50, vit
CONFIG=vitcs_base_16 # swin_tiny_patch4_window7, cvt_13, t2tvit_14, resnet_50, vit_base_16
LAMBDA_DRLOC=0.5 # swin: 0.5, t2t: 0.1, cvt: 0.1
DRLOC_MODE=l1 # l1, ce, cbr
BATCH_SIZE=128
# DATASET=imagenet-100 # imagenet-100, imagenet, cifar-10, cifar-100, svhn, places365, flowers102, clipart, infograph, painting, quickdraw, real, sketch
DATASET=cifar-100
NUM_CLASSES=100


DISK_DATA=${DATA_DIR}/datasets/${DATASET}
# TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-bs${BATCH_SIZE}-choose_scale
# TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-bs${BATCH_SIZE}-choose_trans
# TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-drloc${LAMBDA_DRLOC}-bs${BATCH_SIZE}-choose_scale
# TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-drloc${LAMBDA_DRLOC}-bs${BATCH_SIZE}-choose_trans
TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-drloc${LAMBDA_DRLOC}-bs${BATCH_SIZE}-choose_trans-choose_scale
# TARGET_FOLDER=${DATASET}-${MODE}-sz${IMG_SIZE}-drloc${LAMBDA_DRLOC}-bs128-g1_with_scale
SAVE_DIR=${DATA_DIR}/outputs/SPTv3/visiontransformer-expr/${TARGET_FOLDER}


python3 -m torch.distributed.launch  --master_port 65001 \
    main_affine.py \
    --cfg ./configs/${CONFIG}_${IMG_SIZE}.yaml \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --data-path ${DISK_DATA} \
    --batch-size ${BATCH_SIZE} \
    --output ${SAVE_DIR} \
    --lambda_drloc ${LAMBDA_DRLOC} --drloc_mode ${DRLOC_MODE} --use_drloc  --use_abs --with_trans --with_scale