import os
import pathlib
import shutil

import torch
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms, datasets

from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image




def split_by_labels(string,labels):
  result = []
  for label in labels:
      if label in string:
          result.append(label)
          string = string.replace(label, '')  # Remove the label from the string once found
  return result

def binary_label(present_labels, labels):
  return torch.tensor([1 if label in present_labels else 0 for label in labels])


class MultiLabelDataset(Dataset):
  def __init__(self,root_dir,transforms=None):

    self.root_dir = root_dir
    self.img_paths = list(pathlib.Path(root_dir).glob('*/*.jpg'))
    self.transforms = transforms
    self.classes = [cls.name for cls in list(os.scandir(root_dir))]
    self.class_names = ['Looking Away','Looking at Camera','Looking at Screen','Moving','Smiling','Unlabeled','Upright']


  def __len__(self):
    return len(self.img_paths)

  def class_dict(self):
    class_dict = {}
    for cls in self.classes:
      present_labels = split_by_labels(cls , self.class_names)
      class_dict[cls] = binary_label(present_labels,self.class_names)
    return class_dict

  def __getitem__(self,index:int):
    img = Image.open(self.img_paths[index])
    class_name = self.img_paths[index].parent.name
    class_idx = self.class_dict()[class_name]

    if self.transforms:
      return self.transforms(img) , class_idx
    else:
      return img , class_idx