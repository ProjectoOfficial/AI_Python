from __future__ import print_function

import torch
import torch.nn as nn


#applichiamo la view a mean e std per renderli di dimensione(C * 1 * 1) cosi da riuscire a lavorare direttamente
#con tensori di immagini di shape [B * C * H * W] con B=batch size, C=Channels

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std