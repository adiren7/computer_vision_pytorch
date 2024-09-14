from torch.utils.data import Dataset
from PIL import Image
import os
import pathlib

class BinaryDataset(Dataset):
  def __init__(self,root_dir,transforms=None):

    self.root_dir = root_dir
    self.img_paths = list(pathlib.Path(root_dir).glob('*/*.jpg'))
    self.transforms = transforms
    self.classes = [cls.name for cls in list(os.scandir(root_dir))]


  def __len__(self):
    return len(self.img_paths)


  def class_dict(self):
    return {cls : i for i, cls in enumerate(self.classes)}

  def __getitem__(self,index:int):
    img = Image.open(self.img_paths[index])
    class_name = self.img_paths[index].parent.name
    class_idx = self.class_dict()[class_name]

    if self.transforms:
      return self.transforms(img) , class_idx
    else:
      return img , class_idx