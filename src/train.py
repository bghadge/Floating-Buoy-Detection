import yaml
import os

from torchvision import transforms

from ..utils.dataset import CustomDataset
from ..utils.get_dataloader import get_dataloader
from ..utils.dataset_functions import show_image_batch_with_bboxes

# Get configurations from dataset_config.yaml
pwd = os.path.dirname(os.getcwd())
config_path = os.path.join(pwd, "config/dataset_config.yaml")

with open(config_path) as file:
    config = yaml.full_load(file)

# Create Dataset
transform = transforms.Compose([
                                transforms.Resize(config['reshape_size']),
                                transforms.ToTensor(),
                                transforms.Normalize(config['mean'], config['std'])
                              ])

train_dataset = CustomDataset(config['train'], config['data_dir'], transform)

train_dataloader = get_dataloader(train_dataset, 4)
