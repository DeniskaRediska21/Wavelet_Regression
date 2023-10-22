import cupy as cp
from torchaudio.transforms import FFTConvolve as torch_fftconvolve
from cupyx.scipy.signal import fftconvolve
from numpy import exp
import numpy as np
import torch


class Torch_Morlet_Transform(torch.nn.Module):
    def __init__(self,freq= 1, discretness= 0.01, number_of_scycles =2):
        super().__init__()
        morlet_wavelet = lambda freq,time,number_of_scycles : exp(1j*freq*time)*exp((-time**2*freq**2)/(2*number_of_scycles**2))
        boundaries = (number_of_scycles+1)*3 / freq
        time = np.arange(-boundaries,boundaries,discretness)
        self.wavelet = torch.Tensor(np.real(np.array(np.squeeze([morlet_wavelet(freq,time,number_of_scycles)])))).cuda()


    def forward(self, signal):
        convolve =torch_fftconvolve('valid') 
        convolution =convolve(signal,self.wavelet)

        return convolution



class Torch_Morlet_Transform_imag(torch.nn.Module):
    def __init__(self,freq= 1, discretness= 0.01, number_of_scycles =2):
        super().__init__()
        morlet_wavelet = lambda freq,time,number_of_scycles : exp(1j*freq*time)*exp((-time**2*freq**2)/(2*number_of_scycles**2))
        boundaries = (number_of_scycles+1)*3 / freq
        time = np.arange(-boundaries,boundaries,discretness)
        self.wavelet = torch.Tensor(np.imag(np.array(np.squeeze([morlet_wavelet(freq,time,number_of_scycles)])))).cuda() 


    def forward(self, signal):
        convolve =torch_fftconvolve('valid') 
        convolution =convolve(signal,self.wavelet)

        return convolution


class Morlet_Transform(torch.nn.Module):
    def __init__(self,freq= 1, discretness= 0.01, number_of_scycles =2):
        super().__init__()
        morlet_wavelet = lambda freq,time,number_of_scycles : exp(1j*freq*time)*exp((-time**2*freq**2)/(2*number_of_scycles**2))
        boundaries = (number_of_scycles+1)*3 / freq
        time = np.arange(-boundaries,boundaries,discretness)
        self.wavelet = cp.real(cp.array(np.squeeze([morlet_wavelet(freq,time,number_of_scycles)]))) 


    def forward(self, signal):
        convolution = fftconvolve(cp.array(signal),self.wavelet,'valid')

        convolution = convolution.get()

        return convolution


class Morlet_Transform_imag(torch.nn.Module):
    def __init__(self,freq= 1, discretness= 0.01, number_of_scycles =2):
        super().__init__()
        morlet_wavelet = lambda freq,time,number_of_scycles : exp(1j*freq*time)*exp((-time**2*freq**2)/(2*number_of_scycles**2))
        boundaries = (number_of_scycles+1)*3 / freq
        time = np.arange(-boundaries,boundaries,discretness)
        self.wavelet = cp.imag(cp.array(np.squeeze([morlet_wavelet(freq,time,number_of_scycles)]))) 


    def forward(self, signal):
        convolution = fftconvolve(cp.array(signal),self.wavelet,'valid')

        convolution = convolution.get()

        return convolution

        
#import matplotlib.pyplot as plt
#model = Morlet_Transform(freq = 10, discretness= 0.01)
#
#signal = np.sin(np.arange(0,100,0.001))
#convolution = model(signal)
#
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(convolution,color='tab:blue')
#plt.show()
