import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

from .layers import *
# from ...utils import parsemodelcfg
from .parse_model_cfg import parse_model_cfg
from .create_modules import create_modules


class YOLOv3Model(nn.Module):
    def __init__(self, cfg, img_size=(416, 416), verbose=False, train=False):
        super(YOLOv3Model, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.net_info, self.module_list, self.routs = \
            create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)
        self.training = train
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        # self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, verbose=False):
        # img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        # if verbose:
        #     print('0', x.shape)
        #     str = ''

        for idx, module in enumerate(self.module_list):
            mod_name = module.__class__.__name__
            if mod_name in ['RouteLayer', 'ShortcutLayer']:  # sum, concat
                # if verbose:
                #     l = [i - 1] + module.layers  # layers
                #     sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                #     str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif mod_name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[idx] else [])
            # if verbose:
            #     print('%g/%g %s -' % (idx, len(self.module_list), name), list(x.shape), str)
            #     str = ''

        if self.training:  # train
            return yolo_out
        else:  # inference or test
            # x, p = zip(*yolo_out)  # inference output, training output
            # x = torch.cat(x, 1)  # cat yolo outputs
            return x


def get_yolo_layers(model):
    yolo_layers = []
    for i, m in enumerate(model.module_list):
        if m.__class__.__name__ == 'YOLOLayer':
            yolo_layers.append(i)
    return yolo_layers
