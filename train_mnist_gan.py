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

#####input#####
b_size = 128
nz = 100
lr = 0.0001
num_epochs = 100
###############


#MNISTのデータロード
mnist_data = datasets.MNIST('../data/MNIST', 
                            transform = transforms.Compose([
                                transforms.ToTensor()
                                ]),
                            download=True)
mnist_data_norm = Mydataset(mnist_data)
dataloader = torch.utils.data.DataLoader(mnist_data_norm,
                                          batch_size=b_size,
                                          shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#損失関数の定義
criterion = nn.BCELoss()

#テスト用に使うランダムなベクトル
fixed_noise = torch.randn(5, nz, device=device)

#教師データのラベル
real_label = 1
fake_label = 0

netD = Discriminator().to(device)
netG = Generator().to(device)

#更新手法
optimizerD = optim.Adam(netD.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr)

# トレーニングループ

# 進捗を記録するためのリスト
img_list = []
G_losses = []
D_losses = []
iters = 0

dir_out = './training_process'
if not os.path.exists(dir_out):
    os.mkdir(dir_out)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device).type(torch.float)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(fake)
            print(fake.shape)

        iters += 1

for i in range(len(img_list)):
    for j in range(img_list[i].shape[0]):
        fig = plt.figure()
        plt.imshow(img_list[i].detach().cpu().numpy()[j],cmap='gray')
        fig.savefig(dir_out+'/'+str(i)+'_'+str(j))
        plt.close()

#学習結果を一旦保存
torch.save(netD.state_dict(), './netD.pth')
torch.save(netG.state_dict(), './netG.pth')
