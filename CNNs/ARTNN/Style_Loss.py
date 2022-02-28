from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

'''lo style loss si comporterà come un layer trasparente nella rete neurale che appunto calcola la style loss di un layer.
per calcolarla utilizziamo una matrice di gram (ottenuta moltiplicando la matrice per la sua trasposta).

la matrice di gram va normalizzata dividendo ogni elemento per il numero totale degli elementi della matrice. questo perchè
 nel caso in cui si avesse una mappa delle feature dimensionalmente molto grande, questa genererebbe valori molto grandi nella
 matrice di gram. questi valori molto grandi potrebbero essere conseguenza di un grande impatto sulla discesa del gradiente da 
 parte dei primi layers'''

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.__gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.__gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def __gram_matrix(self, input):
        #a = batch size
        #b = numero di feature maps
        #(c,d) = dimensioni di una feature map
        a, b, c, d = input.size()

        features = input.view(a * b, c * d)

        G = torch.mm(features, features.t())

        #normalizziamo
        return G.__div__(a * b * c * d)