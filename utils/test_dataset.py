import matplotlib.pyplot as plt

from dataset import CustomDataset
from dataset_functions import show_image_with_bboxes

txt = "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/train.txt"
data = "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/labelled/1440_whitebuoy/"

train_dataset = CustomDataset(txt, data)

fig = plt.figure()

for i in range(len(train_dataset)):
    sample = train_dataset[i]

    print(i, sample['image'].shape, sample['bboxes'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    image = sample['image']
    bboxes = sample['bboxes']
    show_image_with_bboxes(image, bboxes)

    if i == 3:
        plt.show()
        break