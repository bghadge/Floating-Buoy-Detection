import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def show_image_with_bboxes(tensor_img, tensor_bboxes, mean=None, std=None,
                           isSingle=False):
    """ Displays image with bounding boxes"""
    """
    Args:
        img (np.array/tensor): np.array of the image.
        bboxes (np.array/tensor): np.array with each row is in yolo format
                            [label, x, y, width, height]
                            where
                            label = integer representing a class
                            x = x/width of the image
                            y = y/height of the image
                            width = width of bbox/width of the image,
                            height = height of bbox/height of the image
    """
    isImgTensor = torch.is_tensor(tensor_img)
    isBboxTensor = torch.is_tensor(tensor_bboxes)

    if mean and std:
        inv_normalize = get_inv_norm(mean, std)

    if isImgTensor:
        tensor_img = inv_normalize(tensor_img)
        chw_img = tensor_img.numpy()
        c, h, w = chw_img.shape
        img = np.zeros((h, w, c))
        img[:, :, 0] = chw_img[0, :, :]
        img[:, :, 1] = chw_img[1, :, :]
        img[:, :, 2] = chw_img[2, :, :]
    elif not isImgTensor:
        img = tensor_img

    if isBboxTensor:
        chw_bboxes = tensor_bboxes.numpy()
        c, h, w = chw_bboxes.shape
        bboxes = np.zeros((h, w))
        bboxes[:, :] = chw_bboxes[0, :, :]
    elif not isBboxTensor:
        bboxes = tensor_bboxes

    height, width, _ = img.shape
    if isSingle:
        all_x, all_rectW = bboxes[:, 1]*width, bboxes[:, 3]*width
        all_y, all_rectH = bboxes[:, 2]*height, bboxes[:, 4]*height
    else:
        all_x, all_rectW = bboxes[:, 0]*width, bboxes[:, 2]*width
        all_y, all_rectH = bboxes[:, 1]*height, bboxes[:, 3]*height

    # Calculate top left and bottom right coords of all bboxes
    top_left_x = all_x - all_rectW/2
    top_left_y = all_y - all_rectH/2
    bot_right_x = all_x + all_rectW/2
    bot_right_y = all_y + all_rectH/2

    for tx, ty, bx, by in zip(top_left_x, top_left_y,
                              bot_right_x, bot_right_y):
        tx, ty, bx, by = int(tx), int(ty), int(bx), int(by)
        # Create a Rectangle on the image
        img = cv2.rectangle(img, (tx, ty), (bx, by), (255, 0, 0), 1)
    plt.imshow(img)


def show_image_batch_with_bboxes(tensor_img, tensor_bboxes, mean, std):
    """ Displays image with bounding boxes"""
    """
    Args:
        img (torch.tensor):     Batch x C x H x W np.array of the image.
        bboxes (torch.tensor):  Batch x no of boxes x 5
    """
    if not torch.is_tensor(tensor_img) and not torch.is_tensor(tensor_img):
        return

    inv_normalize = get_inv_norm(mean, std)
    for i in range(len(tensor_img)):
        tensor_img[i] = inv_normalize(tensor_img[i])

    bchw_imgs = tensor_img.numpy()
    batch, channels, height, width = bchw_imgs.shape
    imgs = np.zeros((batch, height, width, channels))
    imgs[:, :, :, 0] = bchw_imgs[:, 0, :, :]
    imgs[:, :, :, 1] = bchw_imgs[:, 1, :, :]
    imgs[:, :, :, 2] = bchw_imgs[:, 2, :, :]

    bn5_boxs = tensor_bboxes.numpy()

    for i in range(batch):
        ax = plt.subplot(1, batch, i + 1)
        plt.tight_layout(pad=1, h_pad=0.01, w_pad=0.01)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        show_image_with_bboxes(imgs[i], bn5_boxs[i])


def get_inv_norm(mean, std):
    inv_mean = [-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]]
    inv_std = [1/std[0], 1/std[1], 1/std[2]]
    inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)
    return inv_normalize
