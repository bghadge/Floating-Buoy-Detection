import yaml
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset_functions import show_image_batch_with_bboxes

from dataset import CustomDataset
from get_dataloader import get_dataloader

with open("/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/config/dataset_config.yaml") as file:
    config = yaml.full_load(file)

# Create Dataset
transform = transforms.Compose([
                                transforms.Resize(config['reshape_size']),
                                transforms.ToTensor(),
                                transforms.Normalize(config['mean'],
                                                     config['std'])
                              ])

train_dataset = CustomDataset(config['train'], config['data_dir'], transform)

train_dataloader = get_dataloader(train_dataset, 4)

train_loader_iter = iter(train_dataloader)
img, ann = train_loader_iter.next()

print('img has shape: ', img.shape)
print('ann has shape: ', ann.shape)

plt.style.use('dark_background')
plt.figure()
show_image_batch_with_bboxes(img, ann, config['mean'], config['std'])
plt.axis('off')
plt.ioff()
plt.show()
