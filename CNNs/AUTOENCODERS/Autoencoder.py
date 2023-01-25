import torch
from torch import nn
import torch.utils.data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

import os

PATH = os.path.dirname(os.path.realpath(__file__))
BATCH_SIZE = 2048
VERBOSE = 0
TRAIN = 1
VALIDATE = 1 - TRAIN
SAVE = 0
LAE = 0
MSE = 1
P = 0.4

class LinearAutoencoder(nn.Module):

    def __init__(self, input_dim: int = 784, output_dim: int = 784):
        super(LinearAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # encoder
        self.Encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=512),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(P),
            nn.Linear(in_features=512, out_features=self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.Dropout2d(P)
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, (3, 3)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


def imshow(image, label =None):
    npimg = image.numpy()
    plt.imshow(npimg, cmap='gray')
    if label is not None:
        plt.text(0, 0, label, bbox={'facecolor': 'white', 'pad': 10})
    plt.show()


def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


if __name__ == '__main__':

    transform = transforms.Compose([
            transforms.ToTensor()
        ]
    )
    trainset = torchvision.datasets.FashionMNIST(root=os.path.join(PATH, 'data'), train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.FashionMNIST(root=os.path.join(PATH, 'data'), train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if VERBOSE:
        train_features, train_labels = next(iter(trainloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")

        feature, label = next(iter(trainset))
        imshow(feature[0, :, :].squeeze(), output_label(label))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())

    net = None
    if LAE:
        net = LinearAutoencoder(784, 784).to(device)
    else:
        net = ConvAutoencoder().to(device)

    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
    crit = None
    if MSE:
        crit = nn.MSELoss()
    else:
        crit = nn.CrossEntropyLoss()

    epochs = 50
    running_loss = 0.

    if TRAIN:
        for e in range(epochs):
            for i, data in tqdm(enumerate(trainloader), total=int(len(trainset)/trainloader.batch_size)):
                data, _ = data
                data = data.to(device)
                if LAE:
                    data = data.view(data.size(0), -1)

                opt.zero_grad()
                reconstruction = net(data)
                loss = crit(reconstruction, data)
                loss.backward()
                running_loss += loss.item()
                opt.step()

            train_loss = running_loss/(len(trainloader.dataset) * (e + 1))
            print("Epoch: {} - Running loss: {:.10f} - current loss: {:.4f}".format(e, train_loss, loss.item()))

            if (e + 1) % 10 == 0:
                with torch.no_grad():
                    data, label = next(iter(testloader))
                    data = data.to(device)
                    if LAE:
                        data = data.view(data.size(0), -1)
                    reconstruction = net(data)
                    num_rows = 8
                    both = torch.cat((data.view(BATCH_SIZE, 1, 28, 28)[:8],
                                    reconstruction.view(BATCH_SIZE, 1, 28, 28)[:8]))

                    save_image(both.cpu(), os.path.join(PATH, f"output{e}.png"), nrow=num_rows)
                    output = plt.imread(os.path.join(PATH, f"output{e}.png"))
                    plt.imshow(output)
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close()

        if LAE and SAVE:
            torch.save(net.state_dict(), os.path.join(PATH, "LinearAutoencoder.pth"))
        else:
            torch.save(net.state_dict(), os.path.join(PATH, "ConvAutoencoder.pth"))

    elif VALIDATE:
        if LAE:
            torch.save(net.state_dict(), os.path.join(PATH, "LinearAutoencoder.pth"))
        else:
            torch.save(net.state_dict(), os.path.join(PATH, "ConvAutoencoder.pth"))

        for e in range(epochs):
            net.eval()
            running_loss = 0.0
            with torch.no_grad():
                for i, data in tqdm(enumerate(testloader), total=int(len(testset) / testloader.batch_size)):
                    data, _ = data
                    data = data.to(device)
                    if LAE:
                        data = data.view(data.size(0), -1)

                    reconstruction = net(data)
                    loss = crit(reconstruction, data)
                    running_loss += loss.item()

                    # save the last batch input and output of every epoch
                    if i == int(len(testset) / testloader.batch_size) - 1:
                        num_rows = 8
                        both = torch.cat((data.view(BATCH_SIZE, 1, 28, 28)[:8],
                                          reconstruction.view(BATCH_SIZE, 1, 28, 28)[:8]))
                        save_image(both.cpu(), os.path.join(PATH, f"output{e}.png"), nrow=num_rows)
                        output = plt.imread(os.path.join(PATH, f"output{e}.png"))
                        plt.imshow(output)
                        plt.show()

            val_loss = running_loss / len(testloader.dataset)
