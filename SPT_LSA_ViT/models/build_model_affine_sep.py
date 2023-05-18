from .vit_affine import ViT
# from .swin import SwinTransformer
from .swin_affine import SwinTransformer
from .cait import CaiT
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

    if args.model == "vit":
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size,
                    patch_size=patch_size,
                    num_classes=n_classes,
                    dim=192,
                    mlp_dim_ratio=2,
                    depth=9,
                    heads=12,
                    dim_head=192 // 12,
                    stochastic_depth=args.sd,
                    is_SPT=args.is_SPT,
                    is_LSA=args.is_LSA)
        # model = Auxillary(model, 192, aux_out_dim)
        if aux_out_dim == 0:
            return model
        model = Auxillary(model,
                          192,
                          with_scale=args.with_scale,
                          with_trans=args.with_trans)

    elif args.model == 'cait':
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size,
                     patch_size=patch_size,
                     num_classes=n_classes,
                     stochastic_depth=args.sd,
                     is_LSA=args.is_LSA,
                     is_SPT=args.is_SPT)

    elif args.model == 'swin':

        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4

        model = SwinTransformer(img_size=img_size,
                                window_size=window_size,
                                drop_path_rate=args.sd,
                                patch_size=patch_size,
                                mlp_ratio=mlp_ratio,
                                depths=depths,
                                num_heads=num_heads,
                                num_classes=n_classes,
                                is_SPT=args.is_SPT,
                                is_LSA=args.is_LSA)
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