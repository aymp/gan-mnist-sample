import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import os

from model_mnist_gan import *

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #OMP: Error #15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

netG = Generator()

#学習結果を読み込んで、
netG.load_state_dict(torch.load('./netG.pth', map_location=lambda storage, loc: storage))

#推論モードに設定（Dropout、BN、他の無効化）
netG.eval().to(device)

#ランダムなベクトル
fixed_noise = torch.randn(64, 100, device=device)

fake = netG(fixed_noise)#.detach().cpu()

fig = plt.figure(figsize=(15,15))
for i in range(fake.shape[0]):
    ax = fig.add_subplot(8,8,i+1)
    ax.imshow(fake.detach().cpu().numpy()[i],cmap='gray')
fig.savefig('generated_img.png')

