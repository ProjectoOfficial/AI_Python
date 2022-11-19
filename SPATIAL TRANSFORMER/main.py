# code inspired by: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

from __future__ import print_function
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

plt.ion()   # interactive mode

from spatial_transformer import Net
from utils import *
import os
import argparse
from datetime import datetime

from six.moves import urllib


def main(opt):
    device = None
    if opt.force_cpu:
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = None
    test_loader = None
    if opt.mnist:
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ])), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='.', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    else:
        
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root='.', split='train', download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
                            transforms.Grayscale(num_output_channels=1),
                        ])), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root='.', split='test', download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None)
            ])), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    
    number = 0
    for weights in os.listdir():
        if weights.startswith("spatial_transformer_"):
            number = int(weights.split("_")[2].split(".")[0])

    model_name = "spatial_transformer_{}.pt".format(number)

    if os.path.isfile(model_name) and opt.resume == False:
        model = Net().to(device)
        model.load_state_dict(torch.load(model_name))
        test(model, device, test_loader)

        # Visualize the STN transformation on some input batch
        visualize_stn(model, test_loader, device)

        plt.ioff()
        plt.show()
    else:
        epochs = opt.epochs
        model = Net().to(device)
        if opt.resume and os.path.isfile(model_name):
            model.load_state_dict(torch.load(model_name))

        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=0.0001)

        plotname = datetime.now().strftime("train_%d_%m_%Y__%H_%M")
        x = []
        y = []
        for epoch in range(0, epochs):
            x, y = train(epoch, model, optimizer, device, train_loader, plotname, x, y)
            if opt.test:
                test(model, device, test_loader)
            
        torch.save(model.state_dict(), "spatial_transformer_{}.pt".format(number + 1))

if __name__ == '__main__':
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    parser = argparse.ArgumentParser(prog = 'Spatial Transformer', description = 'Deep geometrical transformations', epilog = 'Ciao')
    parser.add_argument('-bs', '--batch-size', type=int, default=100, help='Training and Testing batch size')
    parser.add_argument('-cpu' ,'--force-cpu', action='store_true',  default=False, help='Force CPU usage during training and testing')
    parser.add_argument('-ep', '--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-mn' ,'--mnist', action='store_true',  default=False, help='True if you want to train and test on mnist')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of workers in dataloader')
    parser.add_argument('-r' ,'--resume', action='store_true',  default=False, help='True if you want to resume training')
    parser.add_argument('-t' ,'--test', action='store_true',  default=False, help='True if you to test after training')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0001, help='Weight decay')
    opt = parser.parse_args()
    main(opt)