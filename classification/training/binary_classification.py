import os
import pathlib
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms, models, datasets

from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

criterion = nn.BCEWithLogitsLoss() # Better without Sigmoid in fc layer

def train_step(model , dataloader , loss_fn , optimizer , device ):
  train_losses = 0.0
  train_accs = 0.0

  model.train()
  for X,y in dataloader:

    X , y = X.to(device) , y.float().to(device)

    y_pred = model(X).squeeze()
    y_proba = torch.sigmoid(y_pred)

    y_label = (y_proba > 0.5).float()

    train_loss = loss_fn(y_pred,y)
    train_acc = (y_label == y).sum().item()/len(y_pred)
    train_losses += train_loss
    train_accs += train_acc

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

  train_losses /= len(dataloader)
  train_accs /= len(dataloader)

  return train_losses.item() , train_accs


def test_step(model , dataloader , loss_fn , device) :

  model.eval()

  test_losses = 0.0
  test_accs = 0.0

  with torch.inference_mode():
      for X,y in dataloader:
        X , y = X.to(device) , y.float().to(device)

        y_pred = model(X).squeeze()
        y_proba = torch.sigmoid(y_pred)

        y_label = (y_proba > 0.5).float()

        test_loss = loss_fn(y_pred,y)
        test_acc = (y_label == y).sum().item()/len(y_pred)
        test_losses += test_loss
        test_accs += test_acc


      test_losses /= len(dataloader)
      test_accs /= len(dataloader)

      return test_losses.item() , test_accs
  

def train(model , train_dataloader , test_dataloader , loss_fn , optimizer , epochs  , device):

  results = {
      'train_loss' : [],
      'test_loss' : [],
      'train_acc' : [],
      'test_acc' : []
  }


  for epoch in range(epochs):
    train_losses , train_accs = train_step(model , train_dataloader , loss_fn , optimizer , device )
    test_losses , test_accs = test_step(model , test_dataloader , loss_fn , device )

    print(f"EPOCH : {epoch} | "
          f"TRAIN LOSS : {train_losses:.4f} | "
          f"TEST LOSS : {test_losses:.4f} | "
          f"TRAIN ACCURACIE : {train_accs:.4f} | "
          f"TEST ACCURACIE : {test_accs:.4f}"
    )


    results["train_loss"].append(train_losses)
    results["test_loss"].append(test_losses)
    results["train_acc"].append(train_accs)
    results["test_acc"].append(test_accs)

  return results  