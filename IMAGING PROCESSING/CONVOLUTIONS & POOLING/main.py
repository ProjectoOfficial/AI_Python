import random
import torch
from torch.nn import functional as F

MIN_BATCH = 1
MAX_BATCH = 4

MIN_CHANNELS = 2
MAX_CHANNELS = 4

MIN_KERNEL = 2
MAX_KERNEL = 5

MIN_DIM = 10
MAX_DIM = 20

MIN_STRIDE = 1
MAX_STRIDE = 4


def get_2d_sample():
    n = random.randint(MIN_BATCH, MAX_BATCH)
    iC = random.randint(MIN_CHANNELS, MAX_CHANNELS)
    oC = random.randint(MIN_CHANNELS, MAX_CHANNELS)
    H = random.randint(MIN_DIM, MAX_DIM)
    W = random.randint(MIN_DIM, MAX_DIM)
    kH = random.randint(MIN_KERNEL, MAX_KERNEL)
    kW = random.randint(MIN_KERNEL, MAX_KERNEL)

    s = random.randint(MIN_STRIDE, MAX_STRIDE)

    input = torch.rand(n, iC, H, W, dtype=torch.float32)
    kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)

    return input, kernel, s

def get_3d_sample():
    n = random.randint(MIN_BATCH, MAX_BATCH)
    iC = random.randint(MIN_CHANNELS, MAX_CHANNELS)
    oC = random.randint(MIN_CHANNELS, MAX_CHANNELS)
    T = random.randint(MIN_DIM, MAX_DIM)
    H = random.randint(MIN_DIM, MAX_DIM)
    W = random.randint(MIN_DIM, MAX_DIM)
    kT = random.randint(MIN_KERNEL, MAX_KERNEL)
    kH = random.randint(MIN_KERNEL, MAX_KERNEL)
    kW = random.randint(MIN_KERNEL, MAX_KERNEL)
    
    s = random.randint(MIN_STRIDE, MAX_STRIDE)

    input = torch.rand(n, iC, T, H, W)
    kernel = torch.rand(oC, iC, kT, kH, kW)
    bias = torch.rand(oC)

    return input, kernel, bias, s


def Conv2d(input: torch.Tensor, kernel: torch.Tensor):
    ''' 
    conv2d implementation with stride 1 fixed and no bias
    '''
    [n, iC, H, W] = input.shape
    [oC, iC, kH, kW] = kernel.shape

    out = torch.zeros((n, oC, H-kH+1, W-kW+1), dtype=torch.float32)
    [n, oC, Ho, Wo] = out.shape

    for i in range(Ho):
        for j in range(Wo):
            out[:, :, i, j] = (torch.mul(
                torch.unsqueeze(kernel[:, :, :, :], dim=0),
                torch.unsqueeze(input[:, :, i:i+kH, j:j+kW], dim=1))
                                ).sum(dim=(2,3,4))
    return out

def Conv3d(input: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor):
    ''' 
    conv3d implementation with stride 1 fixed and custom bias
    '''
    [n, iC, T, H, W] = input.shape
    [oC, iC, KT, KH, KW] = kernel.shape

    out = torch.zeros((n, oC, T-KT+1, H-KH+1, W-KW+1), dtype=torch.float32)
    [n, oC, To, Ho, Wo] = out.shape

    for t in range(To):
        for i in range(Ho):
            for j in range(Wo):
                out[:, :, t, i, j] = (torch.mul(
                    torch.unsqueeze(kernel, dim=0),
                    torch.unsqueeze(input[:, :, t:t+KT, i:i+KH, j:j+KW], dim=1))
                                    ).sum(dim=(2, 3, 4, 5)) + bias
    
    return out


def MaxPool2d(input: torch.Tensor, kernel_size: tuple, s: int=1):
    '''
    maxpool2d implementation with stride
    '''
    
    [n, iC, H, W] = input.shape
    kH, kW = kernel_size

    out = torch.zeros((n, iC, (H-kH)//s +1, (W-kW)//s +1), dtype=torch.float32)
    [n, iC, Ho, Wo] = out.shape

    for i in range(0, Ho):
        for j in range(0, Wo):
            inp = input[:, :, i*s:i*s+kH, j*s:j*s+kW]
            out[:, :,i, j] = torch.amax(inp, dim=(2,3))

    return out

def MaxPool3d(input: torch.Tensor, kernel_size: tuple, s: int=1):
    '''
    maxpool3d implementation with stride
    '''
    [n, iC, T, H, W] = input.shape
    kT, kH, kW = kernel_size

    out = torch.zeros((n, iC, (T - kT)//s + 1, (H - kH)//s + 1, (W - kW)//s + 1), dtype=torch.float32)
    [n, iC, To, Ho, Wo] = out.shape

    for t in range(To):
        for i in range(Ho):
            for j in range(Wo):
                inp = input[:, :, t*s:t*s+kT, i*s:i*s+kH, j*s:j*s+kW]
                out[:, :, t, i, j] = torch.amax(inp, dim=(2, 3, 4))
    
    return out

if __name__ == '__main__':
    input, kernel, s = get_2d_sample()
    conv_out = Conv2d(input, kernel)
    # print(conv_out.round() == F.conv2d(input, kernel, stride=1, padding=0).round())

    k = tuple(kernel.shape[2:])
    pool_out = MaxPool2d(input, k, s)
    # print(pool_out == F.max_pool2d(input, k, stride=s))


    input, kernel, bias, s = get_3d_sample()
    conv_out = Conv3d(input, kernel, bias)
    # print(conv_out.round() == F.conv3d(input, kernel, bias, stride=1, padding=0).round())

    k = tuple(kernel.shape[2:])
    pool_out = MaxPool3d(input, k, s)
    # print(pool_out == F.max_pool3d(input, k, stride=s))





