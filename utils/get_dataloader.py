import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CustomDataset

def get_dataloaders(txt_path, data_dir, input_size, batch_size, shuffle = True):
    '''
    Build dataloaders with transformations. 

    Args:
        input_size: int or tuple, the size of the tranformed images
        batch_size: int, minibatch size for dataloading

    Returns:
        dataloader: pytorch dataloader for corresponding txt file.
    '''

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # train_path = './data_miniplaces_modified/train/'
    # test_path = './data_miniplaces_modified/test/'
    # val_path = './data_miniplaces_modified/val/'

    # ========= Step 1: build transformations for the dataset ===========
    # I. Resize the image to input_size using transforms.Resize
    # II. Convert the image to PyTorch tensor using transforms.ToTensor
    # III. Normalize the images with the mean and std parameters of the dataset using transforms.Normalize.

    data_transform = transforms.Compose([
                                          transforms.Resize(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean,std)
                                        ])
    

    # ========= Step 2: build dataloaders for the downloaded data ===========
    # I. Construct pytorch datasets for train/val/test from given txt file path
    dataset = CustomDataset(txt_path, data_dir, data_transform)

    # II. Use torch.utils.data.DataLoader to build dataloaders with the constructed pytorch datasets.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader