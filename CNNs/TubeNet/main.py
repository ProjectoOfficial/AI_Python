import sys
import os

import argparse
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

from tqdm import tqdm

curdir = os.path.dirname(os.path.realpath(__file__))

class TubeSet(Dataset):
    def __init__(self, path: str, normalize: bool=False, scale: bool=False, imshape:tuple=(480, 640)) -> None:
        super(TubeSet, self).__init__()

        if os.path.exists(os.path.join(path, "data", "labels", "labels.csv")):
            self.df = pd.read_csv(os.path.join(path, "data", "labels", "labels.csv"), sep=",")
        else:
            return

        self.imgpath = os.path.join(path, "data", "images")

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(imshape)
        ])

        if normalize and not scale:
            self.mean = float(self.df["views"].mean())
            self.std = float(self.df["views"].std())
            self.df["views"] = (self.df["views"] - self.mean)/self.std
        elif scale and not normalize:
            self.max = float(self.df["views"].max())
            self.min = float(self.df["views"].min())
            self.df["views"] = (self.df["views"] - self.min)/(self.max - self.min)        

    def __getitem__(self, index):
        row = self.df.iloc[index].tolist()
        img = Image.open(os.path.join(self.imgpath, row[0]))
        img = self.transforms(img)

        label = torch.Tensor([float(row[2])])
        return img, label  

    def __len__(self):
        return len(self.df)  


EPOCHS = 100
BATCH_SIZE = 8
RESUME = True
NORMALIZE = True
device = "cuda" if torch.cuda.is_available() else "cpu"

trainset = TubeSet(curdir, normalize=NORMALIZE)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
it = iter(trainset)
img, row = next(it)
plt.imshow(img.permute(1, 2, 0))
plt.show()


model = None
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
if RESUME or not os.path.exists(os.path.join(curdir, "tubenet.pt")):
    if RESUME and os.path.exists(os.path.join(curdir, "tubenet.pt")):
        model.load_state_dict(torch.load(os.path.join(curdir, "tubenet.pt")))
    
    model = model.to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    pbar = tqdm(range(EPOCHS), desc="[TRAINING] loss: NaN, pred: Nan, target: Nan")

    for e in pbar:
        model.train(True)
        for img, target in trainloader:
            img = img.to(device)
            target = target.to(device)
            
            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred, target)
            loss.backward()
            opt.step()
                
        pbar.set_description("[TRAINING] epoch: {}, loss: {:.5f}, pred: {:.5f}, target: {:.5f}".format(e, loss.item(), pred[0].item(), target[0].item()))
    torch.save(model.state_dict(), os.path.join(curdir, "tubenet.pt"))
else:
    model.load_state_dict(torch.load(os.path.join(curdir, "tubenet.pt")))
    model = model.to(device)

#EVAL
count = 0
original, target = next(it)
original, target = next(it)
original, target = next(it)
img = original.to(device)
target = target.to(device)

pred = model(torch.unsqueeze(img, 0))

pred_views = int((pred[0].item() * trainset.std) + trainset.mean)
target_views = int((target[0].item() * trainset.std) + trainset.mean)

plt.imshow(original.permute(1, 2, 0))
plt.title("PREDICTED VIEWS: {} - GT: {}".format(pred_views, target_views))
plt.show()
