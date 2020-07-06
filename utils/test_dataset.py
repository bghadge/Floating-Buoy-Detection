import yaml
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset import CustomDataset
from dataset_functions import show_image_with_bboxes


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

sample = train_dataset[1]
image = sample[0]       # tensor
bboxes = sample[1]      # np.array

plt.style.use('dark_background')
fig = plt.figure()
show_image_with_bboxes(image, bboxes, config['mean'], config['std'], True)
plt.show()
