# import torch
import torch.nn as nn
import numpy as np

from .layers import RouteLayer, ShortcutLayer, YOLOLayer


def create_modules(mod_defs, img_size=(416, 416)):
    net_info = mod_defs[0]     # hyperparameters
    output_filters = [3]    # input image 3 channels
    module_list = nn.ModuleList()
    routs = []  # to keep track of skip connections
    yolo_index = -1

    for index, mod in enumerate(mod_defs[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list
        if (mod["type"] == "convolutional"):
            # Get the info about the layer
            try:
                batch_normalize = int(mod["batch_normalize"])
                bias = False
            except KeyError:
                batch_normalize = 0
                bias = True

            filters = int(mod["filters"])
            kernel_size = int(mod["size"])
            stride = int(mod["stride"])
            padding = int(mod["pad"])
            activation = mod["activation"]

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(output_filters[-1], filters, kernel_size, stride,
                             pad, bias=bias)
            module.add_module('Conv2d', conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('BatchNorm2d', bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('activation', activn)

        # If it's an upsampling layer
        # We use Bilinear2dUpsampling
        elif (mod["type"] == "upsample"):
            stride = int(mod["stride"])
            module = nn.Upsample(scale_factor=stride)
            # module.add_module("upsample_{}".format(index), upsample)

        # If it is a route layer
        elif (mod["type"] == "route"):
            # Convert string of layer numbers to list
            mod["layers"] = [int(x) for x in mod["layers"].split(',')]
            layers = mod['layers']
            filters = sum([output_filters[lyr_idx + 1
                           if lyr_idx > 0 else lyr_idx]
                           for lyr_idx in layers])
            routs.extend([index + lyr_idx
                         if lyr_idx < 0 else lyr_idx
                         for lyr_idx in layers])
            module = RouteLayer(layers=layers)
            # module.add_module("route_{0}".format(index), route)

        # shortcut corresponds to skip connection
        elif mod["type"] == "shortcut":
            prev_layer = int(mod['from'])
            filters = output_filters[-1]
            routs.extend([index + prev_layer])
            module = ShortcutLayer(prev_layer)
            # module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif mod["type"] == "yolo":
            mod['mask'] = mod["mask"].split(",")
            mask = [int(x) for x in mod['mask']]

            anchors = mod["anchors"].split(",")
            anchors = [float(a) for a in anchors]
            anchors = np.array(anchors).reshape((-1, 2))

            yolo_index += 1
            stride = [32, 16, 8]
            module = YOLOLayer(anchors=anchors[mask],  # anchor list
                               nc=int(mod['classes']),  # number of classes
                               img_size=img_size,  # (416, 416)
                               yolo_index=yolo_index,  # 0, 1, 2...
                               stride=stride[yolo_index])

        module_list.append(module)
        output_filters.append(filters)

    routs_binary = [False] * len(mod_defs)
    for i in routs:
        routs_binary[i] = True
    return net_info, module_list, routs_binary
