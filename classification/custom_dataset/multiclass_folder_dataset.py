
import os
import pathlib

from PIL import Image
from torch.utils.data import DataLoader , Dataset


class MulticlassDataset(Dataset):
  def __init__(self,root_dir,transforms):
    self.root_dir = root_dir
    self.img_paths = list(pathlib.Path(root_dir).glob('*/*jpg'))
    self.classes = [cls.name for cls in os.scandir(root_dir)]
    self.class_dict = {cls : i for i , cls in enumerate(self.classes)}

    self.transforms = transforms

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self,index:int):
    img_path = self.img_paths[index] 
    img = Image.open(img_path)
    cls_name = img_path.parent.name
    cls_label = self.class_dict[cls_name]

    if self.transforms:
      return self.transforms(img) , cls_label
    else:
      return img , cls_label