import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CustomDataset


def yolo_collate_fn(batch_lst):
    batch_size = len(batch_lst)
    c, h, w = batch_lst[0][0].shape
    img_batch = torch.zeros(batch_size, 3, h, w)

    max_num_box = max(len(batch_lst[i][1]) \
                      for i in range(batch_size))

    box_batch = torch.Tensor(batch_size, max_num_box, 5).fill_(-1.)
    
    for i in range(batch_size):
      img, ann = batch_lst[i]
      img_batch[i] = img
      all_bbox = ann
      for bbox_idx, one_bbox in enumerate(all_bbox):
        bbox = one_bbox[1:]
        obj_cls = one_bbox[0]
        box_batch[i][bbox_idx] = torch.tensor(np.append(bbox, obj_cls))
    
    return img_batch, box_batch

def get_dataloader(dataset, batch_size, shuffle = True, num_workers=2):
    '''
    Build dataloaders with transformations. 

    Args:
        dataset: pytorch dataset object transformed is preferred
        batch_size: int, minibatch size for dataloading

    Returns:
        dataloader: pytorch dataloader for corresponding txt file.
    '''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=yolo_collate_fn)
    
    return dataloader
