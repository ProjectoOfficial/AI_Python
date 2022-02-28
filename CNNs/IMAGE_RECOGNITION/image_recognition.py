import torch.utils.data
import torchvision
from torchvision import transforms
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

VERBOSE = False
TRAIN = False
PATH = r"C:\Users\daniel\PycharmProjects\Python2021\CNNs\IMAGE_RECOGNITION"
BATCH_SIZE = 2048
RETRAIN = False


class ImageRecognizer(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(ImageRecognizer, self).__init__()

        assert in_channels != 0 and num_classes != 0

        self.AvgPool = nn.AdaptiveAvgPool2d((8, 8))

        self.ConvNN = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, padding_mode='circular'),
            nn.Conv2d(16, 32, 3, padding=1, padding_mode='circular'),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, padding=1, padding_mode='circular'),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='circular'),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, padding=1, padding_mode='circular'),
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='circular'),
        )

        self.LinearNN = nn.Sequential(
            nn.Linear(8*8*128, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.ConvNN(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.LinearNN(x)
        return x


def imshow(image, label):
    img = image * np.array(std)[:, None, None] + np.array(mean)[:, None, None]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if label is not None:
        plt.text(0, 0, label, bbox={'facecolor': 'white', 'pad': 10})
    plt.show()


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
    std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=PATH + r"\data",
        train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=PATH + r"\data",
        train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if VERBOSE:
        train_features, train_labels = next(iter(trainloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0, :, :].squeeze()
        label = train_labels[0]
        imshow(img, label)

        images = train_features[10:20, :, :].squeeze()
        imshow(torchvision.utils.make_grid(images), None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.get_device_name(device))

    net = ImageRecognizer(3, 10).to(device)
    if RETRAIN:
        net.load_state_dict(torch.load(PATH + r"\ImageRecognizerModel.pth"))
    else:
        initialize_weights(net)

    epochs = 10
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001, amsgrad=True)

    if TRAIN:
        for e in range(epochs):
            pbar = tqdm(total=len(trainloader), desc="Epoch {} - 0%".format(e))

            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(device), y.to(device)

                net.zero_grad()
                val = net(x)
                loss = crit(val, y)
                loss.backward()
                opt.step()

                pbar.update(1)
                pbar.set_description("Epoch {} - {:.2f}% -- loss {:.4f}".format(e, i/len(trainloader)*100, loss.item()))

            corr = 0
            with torch.no_grad():
                counter = 0
                for x, y in testloader:
                    x, y = x.to(device), y.to(device)
                    y_pred = net(x)
                    corr += (torch.argmax(y_pred, dim=1) == y).sum().item()
                    counter += BATCH_SIZE
                print("accuracy for Epoch {}: {:.3f}%".format(e, corr/counter*100))

        torch.save(net.state_dict(), PATH + r"\ImageRecognizerModel2.pth")
    else:
        net.load_state_dict(torch.load(PATH + r"\ImageRecognizerModel.pth"))

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        total_params = sum(p.numel() for p in net.parameters())
        print("number of parameters: {}\n".format(total_params))
        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                x, y = data
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(y, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')