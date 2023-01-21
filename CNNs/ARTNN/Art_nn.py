from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

from PIL import Image

import torchvision.transforms as transform
import torchvision.models as models

import copy
import datetime

import Content_Loss
import Normalization
import Style_Loss

#la rete neurale che andremo ad implementare sarà una VGG network, ovvero una tipologia di CNN

class ArtNN():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Cuda available") if torch.cuda.is_available() else print("cpu")

        self.imsize = 512 if torch.cuda.is_available() else 128

        self.loader = transform.Compose([transform.Resize(self.imsize), transform.ToTensor()])

        self.content_img = None
        self.style_img = None

        self.unloader = transform.ToPILImage()

        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()


        #ogni canale è normalizzato dalla media e dalla deviazione standard seguenti
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)

        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def load_images(self, im_ref, im_target):
        self.style_img = self.__image_loader(im_ref)
        self.content_img = self.__image_loader(im_target)

        try:
            assert self.style_img.size() == self.content_img.size()
        except AssertionError:
            print("we need to import style and content images of the same size")

    def __image_loader(self, image_name):
        image = Image.open(image_name)
        image = image.resize((self.imsize, self.imsize), Image.ANTIALIAS)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def __show_tensor_image(self, tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if title is not None:
            cv2.imshow(title, image)
        else:
            cv2.imshow("", image)

    def save_image(self, path, tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if title is not None:
            print(r"{}/{}.jpg".format(path, title), image)
            cv2.imwrite(r"{}/{}.jpg".format(path, title), image)
        else:
            print(r"{}/{}.jpg".format(path, str(datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))))
            cv2.imwrite(r"{}/{}.jpg".format(path, str(datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S"))), image)

    def __get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=None,
                                   style_layers=None):
        if content_layers is None:
            content_layers = self.content_layers_default

        if style_layers is None:
            style_layers = self.style_layers_default

        cnn = copy.deepcopy(cnn)


        normalization = Normalization.Normalization(normalization_mean, normalization_std).to(self.device)

        # questo ci serve per avere degli iteratori a disposizione
        content_losses = []
        style_losses = []

        # assumendo che la cnn sia una nn.Sequential, creiamo una nuova nn.Sequential
        # per andare a inserire i moduli che sono sequenzialmente attivi
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # mettiamo inplace a false perche creerebbe problemi con la style loss e la content loss
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # aggiungiamo una content loss:
                target = model(content_img).detach()
                content_loss = Content_Loss.ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # aggiungiamo una style loss:
                target_feature = model(style_img).detach()
                style_loss = Style_Loss.StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # ora eliminiamo i layers dopo le ultime content loss e style loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], Content_Loss.ContentLoss) or isinstance(model[i], Style_Loss.StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    @staticmethod
    def __get_input_optimizer(input_img):
        # utilizziamo una ottimizzazione numerica
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def __run_style_transfer(self, cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.__get_style_model_and_losses(cnn,
                                                                         normalization_mean, normalization_std,
                                                                         style_img, content_img)
        optimizer = self.__get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correggiamo i valori delle immagini in input
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # un ultima correzione
        input_img.data.clamp_(0, 1)

        return input_img

    def run(self):
        input_img = self.content_img.clone()
        output = self.__run_style_transfer(self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                                    self.content_img, self.style_img, input_img,800)

        return output



