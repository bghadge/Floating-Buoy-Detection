import matplotlib.pyplot as plt
from torchvision import transforms

from dataset import CustomDataset
from dataset_functions import show_image_with_bboxes
from dataset_functions import show_image_batch_with_bboxes

from dataset import CustomDataset
from get_dataloader import get_dataloaders

train_batched_data = get_dataloaders("/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/train.txt",
                                    "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/labelled/1440_whitebuoy/",
                                    (400,600), 2, True)


for i_batch, sample_batched in enumerate(train_batched_data):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['bboxes'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        break
    #     plt.figure()
    #     show_landmarks_batch(sample_batched)
    #     plt.axis('off')
    #     plt.ioff()
    #     plt.show()

# show_image_batch_with_bboxes(batch_img, batch_bboxes)