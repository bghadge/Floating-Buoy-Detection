from ..utils.parse_cfg import parse_cfg
from ..utils.create_modules import create_modules


def test_cfg_crt_mod():
    path = \
        "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/config/yolov3.cfg"
    blocks = parse_cfg(path)
    # print(blocks[0])
    print(create_modules(blocks))
