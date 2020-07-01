import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image_with_bboxes(img, bboxes):
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

# # take these inputs
# txt_file = "train"
# utils_path = "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/utils"

# # May be change finalize dataset_path
# repo_path = os.path.dirname(utils_path)
# txt_path = repo_path + "/dataset/"
# dataset_path = repo_path + "/dataset/labelled/1440_whitebuoy/"

# image_names = pd.read_table(txt_path + txt_file + ".txt", sep="\n", header=None)
# img_name = image_names.iloc[0,0]
# # img = img_name - ".png"
# # print(img)
# print(image_names.iloc[0,0])
# # print('Image name: {}'.format(img_name))
# img = os.path.splitext(img_name)[0]
# img_txt = img + ".txt"
# print(img_txt)


# img_info = pd.read_table(dataset_path + img_txt, sep=" ", header=None)
# # img_info = img_info.split()
# img_info = np.array(img_info)
# img_info = img_info.astype(np.float).reshape(-1, 5)
# print(img_info)

# img_path = dataset_path + img_name

# plt.figure()
# show_image_with_bboxes(img_path, img_info)
# plt.show()
