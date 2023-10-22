import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_transform_module import Torch_Morlet_Transform, Torch_Morlet_Transform_imag
import numpy as np
class Net(nn.Module):
    def __init__(self,freq = 1, discretness= 0.01, number_of_scycles = 2,input_size = 100000):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.morlet_r = Torch_Morlet_Transform(freq = freq, discretness = discretness, number_of_scycles = number_of_scycles) 
        self.morlet_i = Torch_Morlet_Transform_imag(freq = freq, discretness = discretness, number_of_scycles = number_of_scycles) 
        self.linear1 = nn.Linear(2048,512).to(torch.cfloat)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128,1)

        
    # x represents our data
    def forward(self, x):
        WT_i1 = self.morlet_i(x)
        fft_i1 = torch.fft.fft(WT_i1)
        WT_i2 = self.morlet_i(WT_i1)
        fft_i2 = torch.fft.fft(WT_i2)
        
        WT_r1 = self.morlet_r(x)
        fft_r1 = torch.fft.fft(WT_r1)
        WT_r2 = self.morlet_r(WT_r1)
        fft_r2 = torch.fft.fft(WT_r2)
        # (100000-179)*4 +(100000-179*2)*4
        y = self.linear1(torch.cat((WT_i1,fft_i1,WT_i2,fft_i2,WT_r1,fft_r1,WT_r2,fft_r2)))
        shapes = (WT_i1.shape,fft_i1.shape,WT_i2.shape,fft_i2.shape,WT_r1.shape,fft_r1.shape,WT_r2.shape,fft_r2.shape) 
        return shapes



signal = np.sin(np.arange(0,100,0.001))
signal = torch.Tensor(signal).to('cuda:0')
model = Net(freq = 10, discretness= 0.01).to('cuda:0')

print(model(signal))



