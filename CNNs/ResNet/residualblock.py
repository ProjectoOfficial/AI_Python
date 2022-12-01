import torch

class Old_ResidualBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(Old_ResidualBlock, self).__init__()

        self.stride = stride
        self.planes = planes
        self.inplanes = inplanes

        self.conv1 = torch.nn.Conv2d(inplanes, planes, (3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)

        self.conv2 = torch.nn.Conv2d(planes, planes, (3, 3), padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.convG = torch.nn.Conv2d(inplanes, planes, (1, 1), stride=stride, bias=False)
        self.bnG = torch.nn.BatchNorm2d(planes)

        self.relu = torch.nn.ReLU()

    def F(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

    def G(self, x):
        out = x
        if self.stride > 1 or self.inplanes != self.planes:
            out = self.convG(out)
            out = self.bnG(out)
        return out

    def forward(self, x):
        return self.relu(self.F(x) + self.G(x))

class New_ResidualBlock(torch.nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super(New_ResidualBlock, self).__init__()

        self.stride = stride
        self.planes = planes
        self.inplanes = inplanes

        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)

        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.conv3 = torch.nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes)

        self.convG = torch.nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bnG = torch.nn.BatchNorm2d(planes)

        self.relu = torch.nn.ReLU()

    def F(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x

    def G(self, x):
        out = x
        if self.stride > 1 or self.inplanes != self.planes:
            out = self.convG(out)
            out = self.bnG(out)
        return out

    def forward(self, x):
        return self.relu(self.F(x) + self.G(x))