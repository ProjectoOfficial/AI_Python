import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.nn.functional import conv2d
import cv2
import os
from pathlib import Path


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


def run_prev_filters():
    im = load_image(PATH)
    im = torch.from_numpy(im).type(torch.uint8)

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


def sobel(image:torch.Tensor, device: torch.device):
    im = image.to(torch.float32).to(device)
    Sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape((1,1,3,3)).to(device)
    Sy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).reshape((1,1,3,3)).to(device)

    gx = conv2d(im.reshape((1, 1, im.shape[0], im.shape[1])), Sx).to(device)
    gy = conv2d(im.reshape((1, 1, im.shape[0], im.shape[1])), Sy).to(device)

    m = ((gx**2 + gy**2)**(0.5)) / 1140 *255
    d = (torch.atan2(gy, gx)*90/np.pi) + 90
    s = torch.ones(m.shape).to(torch.float32) * 100
    
    image = torch.cat((d,s,m), dim=0).squeeze(1)
    image = cv2.cvtColor(image.numpy().swapaxes(0, 2).swapaxes(0, 1).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image


def live_filtering(filter, params: tuple):
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        frame = np.asarray(frame.mean(axis=2), dtype="float32")

        frame = torch.from_numpy(frame).type(torch.uint8)
        #result = filter(frame, params[0], params[1]).numpy()   # Linear Stretch
        #result = filter(frame, params[0]).numpy()              # Casual treshold

        #t = otsu_treshold(frame, params[0])
        #result = filter(frame, t).numpy()                      # OTSU

        result = filter(frame, params[0])                       # SOBEL
        
        cv2.imshow("Live Filter", result)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("ESC pressed, closing...")
            break

    cam.release()
    cv2.destroyAllWindows()

def image_stitching():
    im_a = cv2.imread("gallery_0.jpg")
    im_a = np.swapaxes(np.swapaxes(im_a, 0, 2), 1, 2)
    im_a = im_a[::-1, :, :]  # from BGR to RGB

    im_b = cv2.imread("gallery_1.jpg")
    im_b = np.swapaxes(np.swapaxes(im_b, 0, 2), 1, 2)
    im_b = im_b[::-1, :, :]  # from BGR to RGB

    points_0 = np.float32([[193, 33], [316, 95], [181, 238], [310, 210]])
    points_1 = np.float32([[137, 50], [339, 58], [132, 191], [335, 199]])

    Tmat = cv2.getPerspectiveTransform(points_1, points_0)

    warp_dst = cv2.warpPerspective(np.swapaxes(np.swapaxes(im_b, 0, 2), 1, 0), Tmat, (int(im_b.shape[2]*0.8), int(im_b.shape[1]*1.8)))

    warp_dst = np.swapaxes(np.swapaxes(warp_dst, 0, 2), 1, 2)

    dim1 = max(warp_dst.shape[1], im_a.shape[1])
    dim2 = max(warp_dst.shape[2], im_a.shape[2])

    out = np.zeros((3, dim1, dim2),dtype=np.uint8)
    out[:, :im_a.shape[1], :] = im_a
    out[:, im_a.shape[1]: warp_dst.shape[1], :warp_dst.shape[2]] = warp_dst[:, im_a.shape[1]: , :]

    plt.imshow(np.swapaxes(np.swapaxes(out, 0, 2), 1, 0))
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    if torch.cuda.is_available():
        print("Actual GPU in use: {}".format(torch.cuda.get_device_name(device))) 
    
    print(Path(__file__).parent.absolute())
    os.chdir(Path(__file__).parent.absolute())
    # run_prev_filters()

    # live_filtering(linear_stretch, (0.8, 10, ))                   # LINEAR TRESHOLD LIVE
    # live_filtering(tresh_filter, (30, ))                          # CASUAL TRESHOLD LIVE
    # live_filtering(tresh_filter, (device, ))                      # OTSU LIVE
    # live_filtering(sobel, (torch.device("cpu"), ))                # SOBEL LIVE

    image_stitching()
    



    

