import matplotlib.pyplot as plt
from torchvision import transforms

from dataset import CustomDataset
from dataset_functions import show_image_with_bboxes

txt = "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/train.txt"
data = "/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/labelled/1440_whitebuoy/"

trans = transforms.Compose([
                            transforms.Resize((320, 224)),
                            transforms.ToTensor(),
                            ])
train_dataset = CustomDataset(txt, data, trans)

fig = plt.figure()

sample = train_dataset[1]
image = sample['image']
bboxes = sample['bboxes']

show_image_with_bboxes(image, bboxes)

plt.show()

# for i in range(len(train_dataset)):
#     sample = train_dataset[i]

#     print(i, sample['image'].shape, sample['bboxes'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')

#     image = sample['image']
#     bboxes = sample['bboxes']

#     show_image_with_bboxes(image, bboxes)

#     if i == 1:
#         plt.show()
        # break