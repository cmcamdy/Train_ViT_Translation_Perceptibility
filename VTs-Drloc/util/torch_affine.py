# ####################### #
# create by cmcandy 2023.3#
# ####################### #
import torchvision.transforms.functional as F

DEBUG = True


class Translate(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitudex, magnitudey, scale=1):
        return F.affine(x,
                        translate=(magnitudex * x.shape[-1],
                                   magnitudey * x.shape[-2]),
                        angle=0,
                        scale=scale,
                        shear=0,
                        fillcolor=self.fillcolor)


class TranslateR(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        # import pdb; pdb.set_trace()
        if DEBUG:
            print("TranslateR", magnitude * x.shape[-2])
        return F.affine(x,
                        translate=(1 * magnitude * x.shape[-1], 0),
                        angle=0,
                        scale=1,
                        shear=0,
                        fillcolor=self.fillcolor)


class TranslateU(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        # import pdb; pdb.set_trace()
        if DEBUG:
            print("TranslateU", magnitude * x.shape[-2])
        return F.affine(x,
                        translate=(0, -1 * magnitude * x.shape[-2]),
                        angle=0,
                        scale=1,
                        shear=0,
                        fillcolor=self.fillcolor)


class TranslateD(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        # import pdb; pdb.set_trace()
        if DEBUG:
            print("TranslateD", magnitude * x.shape[-2])
        return F.affine(x,
                        translate=(0, 1 * magnitude * x.shape[-2]),
                        angle=0,
                        scale=1,
                        shear=0,
                        fillcolor=self.fillcolor)
