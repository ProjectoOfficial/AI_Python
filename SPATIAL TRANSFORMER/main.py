from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()   # interactive mode

from spatial_transformer import Net
from utils import *

from six.moves import urllib

if __name__ == '__main__':
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)


    model = Net().to(device)
    epochs = 20
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, device, train_loader)
        test(model, device, test_loader)

    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.ioff()
    plt.show()

