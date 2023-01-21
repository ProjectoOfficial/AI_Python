import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from time import time

BATCH_SIZE = 1

trainTransform = transforms.Compose([
    transforms.ToTensor()])

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingImages")
print(data_path)

trainData = torchvision.datasets.ImageFolder(root=data_path, transform=trainTransform)

trainLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)

dataIter = iter(trainLoader)

imgs, labels = dataIter.next()

print("images shape: {}".format(imgs.shape))

if 'output' in os.listdir():
    shutil.rmtree('output') 
print("Creating output directory...\n")
os.makedirs('output')

def imsave(imgs,epoch):
    imgs = torchvision.utils.make_grid(imgs)
    npimgs = imgs.numpy()
    plt.figure(figsize=(8, 8))
   # plt.imshow(np.transpose(npimgs, (1, 2, 0)), cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    
    filepath = os.path.join(os.path.dirname(__file__), 'outputs', 'image_epoch_{}.png'.format(epoch))
    plt.imsave(filepath, np.transpose(npimgs, (1, 2, 0)))


Z_dim = 100
H_dim = 128
X_dim = imgs.view(imgs.size(0), -1).size(1)

print("dimensions Z: {}, H: {}, X: {}".format(Z_dim, H_dim, X_dim))

device = 'cuda'


class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Z_dim, H_dim),
            nn.ReLU(),
            nn.Linear(H_dim, X_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


G = Gen().to(device)


class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_dim, H_dim),
            nn.ReLU(),
            nn.Linear(H_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


D = Dis().to(device)

g_opt = opt.Adam(G.parameters(), lr=1e-4)
d_opt = opt.Adam(D.parameters(), lr=1e-6)

for epoch in range(6000):
    start_time=time()
    G_loss_run = 0.0
    D_loss_run = 0.0
    for i, data in enumerate(trainLoader):
        X, _ = data
        X = X.view(X.size(0), -1).to(device)
        mb_size = X.size(0)

        one_labels = torch.ones(mb_size, 1).to(device)
        zero_labels = torch.zeros(mb_size, 1).to(device)

        z = torch.randn(mb_size, Z_dim).to(device)

        D_real = D(X)
        D_fake = D(G(z))

        D_real_loss = F.binary_cross_entropy(D_real, one_labels)
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
        D_loss = D_real_loss + D_fake_loss

        d_opt.zero_grad()
        D_loss.backward()
        d_opt.step()

        z = torch.randn(mb_size, Z_dim).to(device)
        D_fake = D(G(z))
        G_loss = F.binary_cross_entropy(D_fake, one_labels)

        g_opt.zero_grad()
        G_loss.backward()
        g_opt.step()

        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()

    elapsed_time = time() - start_time
    print('Epoch:{},   G_loss:{},    D_loss:{},     time:{:.2f}s'.format(epoch, G_loss_run / (i + 1), D_loss_run / (i + 1), elapsed_time))

    samples = G(z).detach()
    print("sample shape: {}".format(samples.shape))
    samples = samples.view(samples.size(0), 3, 106, 100).cpu()
    imsave(samples, epoch)
