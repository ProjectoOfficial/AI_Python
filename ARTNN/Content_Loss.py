from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

#la funzione content loss rappresenta una versione pesata della distanza di ogni contenuto per ogni layer individuale.
#in questo modo ogni volta che la rete neurale riceve un'immagine in input, viene calcolata la content loss nei layer desiderati
class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        #errore quadratico medio, è sempre non negativo e i valori più vicini allo zero sono migliori
        self.loss = F.mse_loss(input, self.target)
        return input