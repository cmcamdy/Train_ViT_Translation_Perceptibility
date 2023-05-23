# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin import SwinTransformer
from .cvt import CvT
from .t2t import T2t_vit_14
from .resnet import ResNet50
from .vit import ViT
from .vitc import ViTC
from .Auxiliary import Auxillary

def build_modelv2(config):
    model_type = config.MODEL.TYPE
    print(model_type)
    # TODO: finish build model, cvt, twins.....
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                rpe=config.MODEL.SWIN.RPE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                use_drloc=config.TRAIN.USE_DRLOC,
                                sample_size=config.TRAIN.SAMPLE_SIZE,
                                use_multiscale=config.TRAIN.USE_MULTISCALE,
                                drloc_mode=config.TRAIN.DRLOC_MODE,
                                use_abs=config.TRAIN.USE_ABS)
        # model = Auxillary(model, int(config.MODEL.SWIN.EMBED_DIM * 2 ** (len(config.MODEL.SWIN.DEPTHS) - 1)))
        model = Auxillary(model, int(config.MODEL.SWIN.EMBED_DIM * 2 ** (len(config.MODEL.SWIN.DEPTHS) - 1)), with_scale=config.TRAIN.SCALE, with_trans=config.TRAIN.TRANS)
    elif model_type == "cvt":
        model = CvT(num_classes=config.MODEL.NUM_CLASSES,
                    use_drloc=config.TRAIN.USE_DRLOC,
                    sample_size=config.TRAIN.SAMPLE_SIZE,
                    drloc_mode=config.TRAIN.DRLOC_MODE,
                    use_abs=config.TRAIN.USE_ABS)
    elif model_type == "twins":
        model = TwinsSVT(num_classes=config.MODEL.NUM_CLASSES,
                         use_drloc=config.TRAIN.USE_DRLOC,
                         sample_size=config.TRAIN.SAMPLE_SIZE,
                         drloc_mode=config.TRAIN.DRLOC_MODE,
                         use_abs=config.TRAIN.USE_ABS)
    elif model_type == "t2t":
        model = T2t_vit_14(img_size=config.DATA.IMG_SIZE,
                           num_classes=config.MODEL.NUM_CLASSES,
                           use_drloc=config.TRAIN.USE_DRLOC,
                           sample_size=config.TRAIN.SAMPLE_SIZE,
                           drloc_mode=config.TRAIN.DRLOC_MODE,
                           use_abs=config.TRAIN.USE_ABS)
    elif model_type == 'resnet50':
        model = ResNet50(num_classes=config.MODEL.NUM_CLASSES,
                         use_drloc=config.TRAIN.USE_DRLOC,
                         sample_size=config.TRAIN.SAMPLE_SIZE,
                         drloc_mode=config.TRAIN.DRLOC_MODE,
                         use_abs=config.TRAIN.USE_ABS)
    elif model_type == "vit":
        model = ViT(image_size=224,
                    patch_size=16,
                    num_classes=config.MODEL.NUM_CLASSES,
                    dim=768,
                    depth=12,
                    heads=12,
                    mlp_dim=3072,
                    dropout=0.1,
                    emb_dropout=0,
                    use_drloc=config.TRAIN.USE_DRLOC,
                    sample_size=config.TRAIN.SAMPLE_SIZE,
                    drloc_mode=config.TRAIN.DRLOC_MODE,
                    use_abs=config.TRAIN.USE_ABS)
    elif model_type == "vitc":
        model = ViTC(image_size=224,
                     patch_size=16,
                     num_classes=config.MODEL.NUM_CLASSES,
                     dim=768,
                     depth=12,
                     heads=12,
                     mlp_dim=3072,
                     dropout=0.1,
                     emb_dropout=0,
                     use_drloc=config.TRAIN.USE_DRLOC,
                     sample_size=config.TRAIN.SAMPLE_SIZE,
                     drloc_mode=config.TRAIN.DRLOC_MODE,
                     use_abs=config.TRAIN.USE_ABS)
        # model = Auxillary(model, 768)
        model = Auxillary(model, 768, with_scale=config.TRAIN.SCALE, with_trans=config.TRAIN.TRANS)
    elif model_type == "vitcs":
        # import pdb; pdb.set_trace()
        model = ViTC(image_size=32,
                     patch_size=4,
                     num_classes=config.MODEL.NUM_CLASSES,
                     dim=192,
                     depth=9,
                     heads=12,
                     mlp_dim=3072,
                     dropout=0.1,
                     emb_dropout=0,
                     use_drloc=config.TRAIN.USE_DRLOC,
                     sample_size=config.TRAIN.SAMPLE_SIZE,
                     drloc_mode=config.TRAIN.DRLOC_MODE,
                     use_abs=config.TRAIN.USE_ABS)
        # if config.TRAIN.SCALE and config.TRAIN.TRANS:
        #     model = Auxillary(model, 192, 3)
        # elif config.TRAIN.SCALE and not config.TRAIN.TRANS:
        #     model = Auxillary(model, 192, 1)
        # else:
        #     model = Auxillary(model, 192)
        model = Auxillary(model, 192, with_scale=config.TRAIN.SCALE, with_trans=config.TRAIN.TRANS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
