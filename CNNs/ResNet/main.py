import torch
from torch import nn

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import time 
from tqdm import tqdm
import urllib.request
import argparse

from resnet import *

def main(opt):
    # Acquiring device
    device = "cpu"
    if not opt.force_cpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("{}\n".format(device))

    if device == "cuda":
        current = torch.cuda.current_device()
        print(torch.cuda.device(current))
        print(torch.cuda.get_device_name(current))
        print(torch.cuda.get_device_capability(current))
        print(torch.cuda.get_device_properties(current))

    # Testin old residual block
    module = Old_ResidualBlock(256, 128, 1)
    input = torch.zeros((3, 256, 7, 7))
    output = module(input)
    print(output.shape)

    # Testing new residual block
    module = New_ResidualBlock(256, 128, 1).to(device)
    input = torch.zeros((3, 256, 7, 7)).to(device)
    output = module(input)
    print(output.shape)

    # Testing resnet50
    net = ResNet50(opt.classes, opt.channels)

    x = torch.randn((1, opt.channels, opt.classes, opt.classes))
    print(net(x).shape)

    # Downloading ImageNet
    import os
    import shutil

    checkpoints = '/home/imagenet'

    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)

    if not os.path.exists('imagenet64'):
        if not os.path.exists(checkpoints + 'imagenet64.tar'):
            print("Downloading archive...")
            os.chdir(checkpoints)
            urllib.request.urlretrieve("https://pjreddie.com/media/files/imagenet64.tar", filename="imagenet64.tar")
            os.chdir('..')

        print("Copying to local runtime...")
        shutil.copy(checkpoints + 'imagenet64.tar', './imagenet64.tar')
        print("Uncompressing...")
        os.system("tar -xf imagenet64.tar")
    
    print("Data ready!")

    # Loading Dataset
    transform_train = transforms.Compose([
        transforms.Resize(250), 
        transforms.RandomCrop(224, padding=1, padding_mode='edge'), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root='./imagenet64/train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    testset = torchvision.datasets.ImageFolder(root='./imagenet64/val/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # Show some images
    trainiter = iter(trainloader)
    images, labels = trainiter.next()
    images = images[:8]
    print(images.size())

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    imshow(torchvision.utils.make_grid(images))
    print("Labels:" + ' '.join('%9s' % labels[j] for j in range(8)))

    flat = torch.flatten(images, 1)
    print(images.size())
    print(flat.size())

    # Create the model
    model = ResNet50(opt.classes, opt.channels).to(device)

    # Train
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    model.zero_grad()
    model.train()

    for e in range(opt.epochs):
        pbar = tqdm(trainloader, desc="Epoch {} - Batch idx: {} - loss: {}".format(e, 0, "calculating...")) 
        for batch_idx, (x,y) in enumerate(pbar):
            optim.zero_grad()

            x = x.to(device)
            y_true = torch.zeros((opt.batch_size, opt.classes))

            for i in range(len(y)):
                y_true[i][y[i]] = 1
                y_true = y_true.to(device)

            start = time.process_time()
            y_pred = torch.nn.functional.softmax(model(x), dim=1)
            stop = time.process_time()

            loss = crit(y_true, y_pred)
            loss.backward()
            optim.step()

            pbar.set_description(desc="Epoch {} - Batch idx: {} - loss: {:.6f} - runtime: {:.3f}ms".format(e, batch_idx, loss.item(), (stop-start)*1000), refresh=True)

    torch.save(model.state_dict(), "resnet50.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'ResNet', description = 'Image Classification Network', epilog = 'Ciao')
    parser.add_argument('-bs', '--batch-size', type=int, default=256, help='Training and Testing batch size')
    parser.add_argument('-cpu' ,'--force-cpu', action='store_true',  default=False, help='Force CPU usage during training and testing')
    parser.add_argument('-chn', '--channels', type=int, default=3, help='Number of channels in input images')
    parser.add_argument('-cls', '--classes', type=int, default=1000, help='Number of classes')
    parser.add_argument('-ep', '--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of workers in dataloader')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.00001, help='Weight decay')
    opt = parser.parse_args()
    main(opt)
