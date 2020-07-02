import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def show_image_with_bboxes(tensor_img, tensor_bboxes):
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
    if torch.is_tensor(tensor_img) and torch.is_tensor(tensor_bboxes):
        chw_img = tensor_img.numpy()
        print('chw_img shape: ',chw_img.shape)
        c, h, w = chw_img.shape
        img = np.zeros((h, w, c))
        img[:,:,0] = chw_img[0,:,:]
        img[:,:,1] = chw_img[1,:,:]
        img[:,:,2] = chw_img[2,:,:]
        # print("img shape:", img.shape)
        chw_bboxes = tensor_bboxes.numpy()
        c, h, w = chw_bboxes.shape
        bboxes = np.zeros((h, w))
        bboxes[:,:] = chw_bboxes[0,:,:]
        print("bbox shape:", bboxes.shape)
    else:
        img = tensor_img
    height, width, _ = img.shape
    all_x, all_rectW = bboxes[:, 1]*width, bboxes[:, 3]*width
    all_y, all_rectH = bboxes[:, 2]*height, bboxes[:, 4]*height
    # Calculate top left and bottom right coords of all bboxes
    top_left_x = all_x - all_rectW/2
    top_left_y = all_y - all_rectH/2
    bot_right_x = all_x + all_rectW/2
    bot_right_y = all_y + all_rectH/2
    
    for tx,ty,bx,by in zip(top_left_x,top_left_y,bot_right_x,bot_right_y):
        tx,ty,bx,by = int(tx),int(ty),int(bx),int(by)
        # Create a Rectangle on the image
        img = cv2.rectangle(img, (tx,ty), (bx,by), (255,0,0), 1)

    # Display the image
    plt.imshow(img)

def show_image_batch_with_bboxes(tensor_img, bboxes):
    """ Displays image with bounding boxes"""
    """
    Args:
        img (np.array): np.array of the image.
        bboxes (np.array): np.array with each row is in yolo format
                            [label, x, y, width, height]
                            where
                            label = integer representing a class
                            x = x/width of the image
                            y = y/height of the image
                            width = width of bbox/width of the image,
                            height = height of bbox/height of the image
    """
    if torch.is_tensor(tensor_img):
        chw_img = tensor_img.numpy()
        print('chw_img shape: ',chw_img.shape)
        c, h, w = chw_img.shape
        img = np.zeros((h, w, c))
        img[:,:,0] = chw_img[0,:,:]
        img[:,:,1] = chw_img[1,:,:]
        img[:,:,2] = chw_img[2,:,:]
        # print("img shape:", img.shape)    
    else:
        img = tensor_img
    height, width, _ = img.shape
    all_x, all_rectW = bboxes[:, 1]*width, bboxes[:, 3]*width
    all_y, all_rectH = bboxes[:, 2]*height, bboxes[:, 4]*height
    # Calculate top left and bottom right coords of all bboxes
    top_left_x = all_x - all_rectW/2
    top_left_y = all_y - all_rectH/2
    bot_right_x = all_x + all_rectW/2
    bot_right_y = all_y + all_rectH/2
    
    for tx,ty,bx,by in zip(top_left_x,top_left_y,bot_right_x,bot_right_y):
        tx,ty,bx,by = int(tx),int(ty),int(bx),int(by)
        # Create a Rectangle on the image
        img = cv2.rectangle(img, (tx,ty), (bx,by), (255,0,0), 1)

    # Display the image
    plt.imshow(img)
