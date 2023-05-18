# Credits to DeepVoltaire
# github:DeepVoltaire/AutoAugment

import numpy as np
# from util.transforms import *
try:
    from torch_affine import *
except ImportError:
    from .torch_affine import *
from torchvision import transforms
import random
import torch


class TranslatePolicy(object):
    """ 
        Randomly Translate & Get label
    """
    def __init__(self,
                 fillcolor=(128, 128, 128),
                 magnitude=np.linspace(-0.3, 0.3, 20),
                 scale=np.linspace(0.5, 1.5, 10),
                 with_scale=True,
                 with_trans=True):
        self.magnitude = magnitude
        self.scale = scale
        self.trans = Translate(fillcolor)
        self.with_scale = with_scale
        self.with_trans = with_trans

    def __call__(
        self,
        img,
    ):
        # print(self.with_scale, self.with_trans)
        if not self.with_scale and not self.with_trans: return img, None
        img_list = list()
        label_list = list()
        for batch in range(img.shape[0]):
            magnitude_x, magnitude_y, scale = 0, 0, 1
            if self.with_trans:
                magnitude_x = self.magnitude[random.randint(
                    0, self.magnitude.shape[0] - 1)]
                magnitude_y = self.magnitude[random.randint(
                    0, self.magnitude.shape[0] - 1)]
            # import pdb
            # pdb.set_trace()
            if self.with_scale:
                scale = self.scale[random.randint(0, self.scale.shape[0] - 1)]
            magnitude_x_label = self.norm(magnitude_x, self.magnitude[0],
                                          self.magnitude[-1])
            magnitude_y_label = self.norm(magnitude_y, self.magnitude[0],
                                          self.magnitude[-1])
            scale_label = self.norm(scale, self.scale[0], self.scale[-1])
            if self.with_scale and self.with_trans:
                label = torch.tensor(
                    (magnitude_x_label, magnitude_y_label, scale_label),
                    dtype=torch.float).reshape(1, -1).to(img.device)
            if self.with_scale and not self.with_trans:
                label = torch.tensor(
                    (scale_label),
                    dtype=torch.float).reshape(1, -1).to(img.device)
            if not self.with_scale and self.with_trans:
                label = torch.tensor(
                    (magnitude_x_label, magnitude_y_label),
                    dtype=torch.float).reshape(1, -1).to(img.device)
            img_list.append(
                self.trans(img[batch], magnitude_x, magnitude_y, scale))
            label_list.append(label)

        return torch.stack(img_list, dim=0), torch.cat(label_list, dim=0)

    def __repr__(self):
        return "AutoAugment Small Dataset TranslatePolicy"

    def norm(self, data, min_board, max_board):
        '''
            norm to (0,1)
        '''
        return (data - min_board) / (max_board - min_board)


if __name__ == "__main__":

    import torchvision
    from PIL import Image
    import torch
    fillcolor = (128, 128, 128)
    # trans = TranslatePolicy(with_trans=False)
    trans = TranslatePolicy(with_scale=False)

    img_1 = Image.open(
        "/work/workspace12/chenhuan/code/datasets/imagenet/ILSVRC2012_img_train/n01440764/n01440764_18.JPEG"
    )
    img_2 = Image.open(
        "/work/workspace12/chenhuan/code/datasets/imagenet/ILSVRC2012_img_train/n01440764/n01440764_18.JPEG"
    )
    img_1 = torchvision.transforms.ToTensor()(img_1)
    img_2 = torchvision.transforms.ToTensor()(img_2)
    torchvision.utils.save_image(
        img_1, "/work/workspace12/chenhuan/code/outputs/test/sample/img.png")
    img = torch.stack([img_1, img_2], dim=0)
    # img = torch.randn(2, 3, 8, 8).cuda()
    # img = torch.randn(2, 1, 8, 8).cuda()
    print(img)
    for indx in range(10):
        trans_img, label = trans(img)
        print(trans_img, label.shape, label)
        for batch in range(trans_img.shape[0]):
            torchvision.utils.save_image(
                trans_img[batch],
                f"/work/workspace12/chenhuan/code/outputs/test/sample/trans_img_{indx}_{batch}.png"
            )
        # pdb.set_trace()
        # trans_img.save(
        #     "/work/workspace12/chenhuan/code/outputs/test/sample/random_changed_{}_{}.png"
        #     .format(indx, indy))
