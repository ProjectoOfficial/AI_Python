import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int):
        super(ResidualBlock, self).__init__()

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

module = ResidualBlock(256, 128, 2)
input = torch.zeros((3, 256, 7, 7))
output = module(input)
print(output.shape)