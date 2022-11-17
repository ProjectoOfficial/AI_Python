import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, stride = 1, kernel_size=5):
        super(ResidualBlock, self).__init__()

        self.stride = stride
        self.planes = planes
        self.inplanes = inplanes
        self.kernel_size = kernel_size
        self.pad = kernel_size //2

        self.conv1 = torch.nn.Conv2d(inplanes, planes, (1, 1), stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)

        self.conv2 = torch.nn.Conv2d(planes, planes, (self.kernel_size, self.kernel_size), padding=self.pad, stride=self.stride, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.conv3 = torch.nn.Conv2d(planes, planes, (1, 1), stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes)

        self.convG = torch.nn.Conv2d(inplanes, planes, (1, 1), stride=stride, bias=False)
        self.bnG = torch.nn.BatchNorm2d(planes)

        self.relu = torch.nn.ReLU()

    def F(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        return x

    def G(self, x):
        out = x
        if self.stride > 1 or self.inplanes != self.planes:
            out = self.convG(out)
            out = self.bnG(out)
        return out

    def forward(self, x):
        return self.relu(self.F(x) + self.G(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv1 = ResidualBlock(1, 10, stride=1, kernel_size=5)
        self.conv2 = ResidualBlock(10, 20, stride=1, kernel_size=5)
        
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(980, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 980)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
