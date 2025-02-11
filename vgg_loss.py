from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class vgg_loss(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(vgg_loss, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        # self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False
        self.vgg.cuda()
        
    def forward(self, sr, hr):
        def _forward(x):
            x = self.vgg(x)
            return x    
        vgg_sr = _forward(sr.tile(1,3,1,1))
        with torch.no_grad():
            vgg_hr = _forward(hr.tile(1,3,1,1).detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss