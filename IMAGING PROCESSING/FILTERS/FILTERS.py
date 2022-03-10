# @author Dott. Daniel Rossi

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision

PATH = r"C:\Users\daniel\Desktop\Filtri\gatto.jpg"

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

def linear_stretch(image: torch.tensor, a: torch.float32, b: torch.float32):
    return torch.clamp(torch.round((image.to(torch.float32) * a) + b), min=0, max=255).type(torch.uint8)


def otsu_treshold(image: torch.tensor, device: torch.device):
    hist = torch.zeros((256, ))
    for i in range(256):
        hist[i] = torch.sum(image.type(torch.float32) == i)

    tresh = -1
    maxvar = -1
    dat = torch.arange(256).type(torch.float32)

    for i in range(256):
        w1 = hist[:i].sum()
        w2 = hist[i:].sum()

        if w1 == 0 or w2 == 0:
            continue
        m1 = torch.sum(hist[:i]* dat[:i])/w1
        m2 = torch.sum(hist[i:]* dat[i:])/w2
        var = w1*w2*((m1-m2)**2)
        if var > maxvar:
            maxvar = var
            tresh = i
    return tresh - 1


def tresh_filter(image:torch.tensor, treshold: torch.uint8):
    print("Using treshold: {}".format(treshold))
    return ((image > treshold) * 255).type(torch.uint8)

def im_show(image: torch.tensor):
    plt.imshow(image, cmap="gray")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    if torch.cuda.is_available():
        print("Actual GPU in use: {}".format(torch.cuda.get_device_name(device))) 

    im = load_image(PATH)
    im = torch.from_numpy(im).type(torch.uint8).to(device)
    
    # linear stretch
    filtered = linear_stretch(im, 0.8, 5)
    im_show(filtered)

    # casual treshold filter
    treshold = 70
    filtered = tresh_filter(im, treshold)
    im_show(filtered)

    # OTSU treshold filter
    treshold = otsu_treshold(im, device)
    filtered = tresh_filter(im, treshold)
    im_show(filtered)



    

