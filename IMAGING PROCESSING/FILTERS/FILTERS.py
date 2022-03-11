# @author Dott. Daniel Rossi

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision

PATH = r"C:\Users\daniel\Documents\GitHub Repositories\AI_Python\IMAGING PROCESSING\FILTERS\gatto.jpg"

def load_image(infilename):
    """This function loads an image into memory when you give it
       the path of the image
    """
    img = Image.open(infilename)
    img.load()
    trasformer = torchvision.transforms.Grayscale(1)
    img = trasformer(img)
    data = np.asarray(img, dtype="float32")
    return data

def linear_stretch(image: torch.tensor, a: torch.float32, b: torch.float32, device: torch.device):
    return torch.clamp(torch.round((image.to(torch.float32) * a) + b).to(device), min=0, max=255).to(device).type(torch.uint8)


def otsu_treshold(image: torch.tensor, device: torch.device):
    hist = torch.zeros((256, ))
    for i in range(256):
        hist[i] = torch.sum(image.type(torch.float32) == i).to(device)

    tresh = -1
    maxvar = -1
    dat = torch.arange(256).type(torch.float32)

    for i in range(256):
        w1 = hist[:i].sum().to(device)
        w2 = hist[i:].sum().to(device)

        if w1 == 0 or w2 == 0:
            continue
        m1 = torch.sum(hist[:i]* dat[:i]).to(device)/w1
        m2 = torch.sum(hist[i:]* dat[i:]).to(device)/w2
        var = w1*w2*((m1-m2)**2)
        if var > maxvar:
            maxvar = var
            tresh = i

    if tresh > 256:
        return 255
    
    if tresh <=0:
        return 0

    return tresh - 1


def median_treshold(image: torch.tensor, device: torch.device):
    return torch.median(torch.median(image).to(device)).to(device).to(torch.uint8)


def average_treshold(image: torch.tensor, device: torch.device):
    image = image.to(torch.float32)
    return torch.mean(torch.mean(image).to(device)).to(device).to(torch.uint8)

def minmax_treshold(image: torch.tensor, device: torch.device):
    min = torch.min(torch.min(image).to(device)).to(device)
    max = torch.max(torch.max(image).to(device)).to(device)
    return (torch.round((max - min) / 2).to(device)).to(torch.uint8)
 
def tresh_filter(image:torch.tensor, treshold: torch.uint8, device: torch.device, label: str):
    print("{}: {}".format(label, treshold))
    return ((image > treshold).to(device) * 255).type(torch.uint8)

def im_show(image: torch.tensor, label: str):
    image = image.detach().cpu()
    plt.imshow(image, cmap="gray")
    plt.title(label)
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        print("Actual GPU in use: {}".format(torch.cuda.get_device_name(device)))
    else:
        device = "cpu"
    print("Running on device: {}".format(device))

    im = load_image(PATH)
    im = torch.from_numpy(im).type(torch.uint8)
   
    im = im.to(device)
    
    im_show(im, "Original image")

    # linear stretch
    filter_name = "Linear stretch"
    filtered = linear_stretch(im, 1.2, 25, device)
    im_show(filtered, filter_name)

    # casual treshold filter
    filter_name = "Casual treshold"
    treshold = 70
    filtered = tresh_filter(im, treshold, device, filter_name)
    im_show(filtered, filter_name)

    # Average Filter
    filter_name = "Average treshold"
    treshold = average_treshold(im, device)
    filtered = tresh_filter(im, treshold, device, filter_name)
    im_show(filtered, filter_name)

    # Median Filter
    filter_name = "Median treshold"
    treshold = median_treshold(im, device)
    filtered = tresh_filter(im, treshold, device, filter_name)
    im_show(filtered, filter_name)
    
    # MinMax Filter
    filter_name = "MinMax treshold"
    treshold = minmax_treshold(im, device)
    filtered = tresh_filter(im, treshold, device, filter_name)
    im_show(filtered, filter_name)

    # OTSU treshold filter
    filter_name = "OTSU treshold"
    treshold = otsu_treshold(im, device)
    filtered = tresh_filter(im, treshold, device, filter_name)
    im_show(filtered, filter_name)



    

