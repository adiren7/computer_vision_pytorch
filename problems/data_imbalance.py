import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods for dealing with imbalanced datasets:
# 1. Oversampling (probably preferable)
# 2. Class weighting


def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # Add more if you want
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    subdirectories = dataset.classes
    class_weights = []

    # loop through each subdirectory and calculate the class weight
    # that is 1 / len(files) in that subdirectory
    for subdir in subdirectories:
        files = os.listdir(os.path.join(root_dir, subdir))
        class_weights.append(1 / len(files))

    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader
