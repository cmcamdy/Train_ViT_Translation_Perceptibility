from .vit_affine import VisionTransformer
# from .swin import SwinTransformer
from .swin_affine import SwinTransformer
from .cait import cait_models
from functools import partial
from torch import nn
# from .Auxiliary import Auxillary
from .Auxiliary_sep import Auxillary


def create_model(img_size, n_classes, args):

    aux_out_dim = 0
    if args.with_scale and args.with_trans:
        aux_out_dim = 3
    elif args.with_scale and not args.with_trans:
        aux_out_dim = 1
    elif not args.with_scale and args.with_trans:
        aux_out_dim = 2

    if args.arch == "vit":
        patch_size = 4 if img_size == 32 else 8  #4 if img_size = 32 else 8
        model = VisionTransformer(img_size=[img_size],
                                  patch_size=args.patch_size,
                                  in_chans=3,
                                  num_classes=n_classes,
                                  embed_dim=192,
                                  depth=9,
                                  num_heads=12,
                                  mlp_ratio=args.vit_mlp_ratio,
                                  qkv_bias=True,
                                  drop_path_rate=args.sd,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # model = Auxillary(model, 192, aux_out_dim)
        if aux_out_dim == 0:
            return model
        model = Auxillary(model,
                          192,
                          with_scale=args.with_scale,
                          with_trans=args.with_trans)

    elif args.arch == 'cait':
        patch_size = 4 if img_size == 32 else 8
        model = cait_models(img_size=img_size,
                            patch_size=patch_size,
                            embed_dim=192,
                            depth=24,
                            num_heads=4,
                            mlp_ratio=args.vit_mlp_ratio,
                            qkv_bias=True,
                            num_classes=n_classes,
                            drop_path_rate=args.sd,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            init_scale=1e-5,
                            depth_token_only=2)

    elif args.arch == 'swin':

        mlp_ratio = args.vit_mlp_ratio
        if img_size == 32:
            window_size = 4
            patch_size = 2
        elif img_size == 64:
            window_size = 4
            patch_size = 4
        else:
            window_size = 7
            patch_size = args.patch_size

        model = SwinTransformer(img_size=img_size,
                                window_size=window_size,
                                patch_size=patch_size,
                                embed_dim=96,
                                depths=[2, 6, 4],
                                num_heads=[3, 6, 12],
                                num_classes=n_classes,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                drop_path_rate=args.sd)
        if aux_out_dim == 0:
            return model
        model = Auxillary(model,
                          384,
                          with_scale=args.with_scale,
                          with_trans=args.with_trans)
        # import pdb; pdb.set_trace()
    else:
        NotImplementedError("Model architecture not implemented . . .")

    return model