import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import utils.glob as CDglobs
import utils.utils as Utils
import pathlib

train_data = datasets.ImageFolder('session03/data/catsDogs/training_set', transform=CDglobs.train_transforms)
# Utils.imshow(train_data[3201][0])
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

if train_data:
    print('Data found!')
    Utils.imshow(train_data[3201][0])
