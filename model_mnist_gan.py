import torch
import torch.nn as nn
from torch.utils.data import Dataset

#生成器でtanhを使っているため入力の範囲を[-1, 1]にする。
class Mydataset(Dataset):
    def __init__(self, mnist_data):
        self.mnist = mnist_data
    def __len__(self):
        return len(self.mnist)
    def __getitem__(self, idx):
        X = self.mnist[idx][0]
        X = (X * 2) - 1
        y = self.mnist[idx][1]
        
        return X, y

#generatorの定義
class Generator(nn.Module):
    def __init__(self, input_size = 100, hid1_size = 256,
                 hid2_size = 512, hid3_size = 1024, batch_size = 4):
        super(Generator, self).__init__()
        self.b_size = batch_size
        self.fc1 = nn.Linear(input_size, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, 28 * 28 * 1)
    
        self.bn1 = nn.BatchNorm1d(hid1_size)
        self.bn2 = nn.BatchNorm1d(hid2_size)
        self.bn3 = nn.BatchNorm1d(hid3_size)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
    
    def forward(self, input):
        x = input.view(-1, 100)
        x = self.LeakyReLU(self.fc1(x))
        x = self.bn1(x)
        x = self.LeakyReLU(self.fc2(x))
        x = self.bn2(x)
        x = self.LeakyReLU(self.fc3(x))
        x = self.bn3(x)
        x = torch.tanh(self.fc4(x))
        x = x.view(-1, 28, 28)
        return x

#discriminatorの定義
class Discriminator(nn.Module):
    def __init__(self, hid1_size = 1024, hid2_size = 512,
                 hid3_size = 256, batch_size = 4):
        super(Discriminator, self).__init__()
        self.b_size = batch_size
        self.fc1 = nn.Linear(784, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, 1)
    
        self.LeakyReLU = nn.LeakyReLU(0.2)
    
    def forward(self, input):
        x = input.view(-1, 784)
        x = self.LeakyReLU(self.fc1(x))
        x = self.LeakyReLU(self.fc2(x))
        x = self.LeakyReLU(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x