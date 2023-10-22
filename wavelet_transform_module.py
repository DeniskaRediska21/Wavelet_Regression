import cupy as cp
from cupyx.scipy.signal import fftconvolve
from numpy import exp
import numpy as np
import torch

class Morlet_Transform(torch.nn.Module):
    def __init__(self,freq= 1, discreteness= 0.01, number_of_scycles =2):
        super().__init__()
        morlet_wavelet = lambda freq,time,number_of_scycles : exp(1j*freq*time)*exp((-time**2*freq**2)/(2*number_of_scycles**2))
        boundaries = (number_of_scycles+1)*3 / freq
        time = np.arange(-boundaries,boundaries,discreteness)
        self.wavelet = cp.real(cp.array(np.squeeze([morlet_wavelet(freq,time,number_of_scycles)]))) 


    def forward(self, signal):
        convolution = fftconvolve(cp.array(signal),self.wavelet,'valid')

        convolution = convolution.get()

        return convolution


        
#import matplotlib.pyplot as plt
#model = Morlet_Transform(freq = 10, discreteness= 0.01)
#
#signal = np.sin(np.arange(0,100,0.001))
#convolution = model(signal)
#
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(convolution,color='tab:blue')
#plt.show()
