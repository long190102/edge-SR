import cv2
import numpy as np
import os
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torchvision.utils import save_image
import torchsummary
from tqdm import tqdm
from math import sqrt

import time




class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class edgeVDSR(nn.Module):
    def __init__(self,num_channels = 3, edge_option = 'sobel'):
        super(edgeVDSR, self).__init__()
        self.edgeoption = edge_option
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 16)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.edgelayer = EFE(num_channels)
        self.sobel = SobelFilterOpenCV()
        self.canny = CannyFilterOpenCV()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.edgeoption =='sobel':
            y = self.sobel(x)
        if self.edgeoption =='canny':
            y = self.canny(x)
        residual = x
        res1 = self.edgelayer(y)
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = out + res1
        out = self.output(out)
        out = out + residual
        return out


class EFE(nn.Module):
    def __init__(self, num_channels = 3):
        
        super(EFE, self).__init__()
        self.path1 = nn.Sequential(
            #nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None, recompute_scale_factor=None),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            #nn.Upsample(size=None, scale_factor=4, mode='bicubic', align_corners=None, recompute_scale_factor=None)
        )
        self.out = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fuse = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        #lecun_normal_(self.path1[2].weight)
        #lecun_normal_(self.fuse.weight)
        #nn.init.zeros_(self.path1[2].bias)
        #nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        a = self.path1(x)
        b = self.path2(x)
        out = a * b
        out = self.out(out)
        out = self.fuse(out + b)
        return out 
    
class CannyFilterOpenCV(nn.Module):
    def __init__(self, low_threshold=0.1*255, high_threshold=0.3*255):
        super(CannyFilterOpenCV, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, x):
        x_np = x.cpu().detach().numpy()
        canny_edges_batch = []
        for img in x_np:
            img_np = img.transpose(1, 2, 0)
            img_np = np.uint8(img_np * 255)
            canny_edges = cv2.Canny(img_np, self.low_threshold, self.high_threshold) # type: ignore
            #canny_edges = cv2.Canny(lr_image, 0.1*255, 0.3*255)
            canny_edges = cv2.cvtColor(canny_edges,cv2.COLOR_GRAY2RGB)
            canny_edges = canny_edges / 255.0
            canny_edges_batch.append(canny_edges.transpose(2, 0, 1))
        canny_edges_tensor = torch.from_numpy(np.array(canny_edges_batch)).float().to(x.device)
        #print(canny_edges_tensor.shape)
        return canny_edges_tensor
    
    def get_output_channels(self):
        
        return 3 


class SobelFilterOpenCV(nn.Module):
    def __init__(self):
        super(SobelFilterOpenCV, self).__init__()

    def forward(self, x):
        x_np = x.cpu().detach().numpy()
        sobel_edges_batch = []
        for img in x_np:
            img_np = img.transpose(1, 2, 0)
            img_np = np.uint8(img_np * 255)
            sobel_x = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3) # type: ignore
            sobel_y = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3) # type: ignore
            sobel_edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            sobel_edges = cv2.normalize(sobel_edges, None, 0, 1, cv2.NORM_MINMAX) # type: ignore
            sobel_edges_batch.append(sobel_edges.transpose(2, 0, 1))
        sobel_edges_tensor = torch.from_numpy(np.array(sobel_edges_batch)).float().to(x.device)
        return sobel_edges_tensor

    def get_output_channels(self, input_channels):
        return input_channels
