# import os
# import numpy as np


def parse_model_cfg(cfg_path):
    """
    Takes a configuration file
    input: .cfg file path

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    output: blocks = {params, layer1, layer2,...}
    """
    file = open(cfg_path, 'r')
    # store the lines in a list
    lines = file.read().split('\n')
    # get read of the empty lines
    lines = [x for x in lines if len(x) > 0]
    # get rid of comments
    lines = [x for x in lines if x[0] != '#']
    # get rid of fringe whitespaces
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":            # This marks the start of a new block
            # If block is not empty, implies it is
            # storing values of previous block.
            if len(block) != 0:
                blocks.append(block)  # add it the blocks list
                block = {}            # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
