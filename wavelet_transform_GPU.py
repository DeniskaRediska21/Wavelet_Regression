import cupy as cp
from cupyx.scipy.signal import fftconvolve
from numpy import exp
import matplotlib.pyplot as plt
import numpy as np
import timeit   

def wavelet_transform(signal, freq=1, discreteness=0.01,number_of_scycles=2,verbose = False):
    morlet_wavelet = lambda freq,time,number_of_scycles : exp(1j*freq*time)*exp((-time**2*freq**2)/(2*number_of_scycles**2))

    boundaries = (number_of_scycles+1)*3 / freq
    time = np.arange(-boundaries,boundaries,discreteness)


    wavelet = np.squeeze([morlet_wavelet(freq,time,number_of_scycles)]) 

    convolution = fftconvolve(cp.array(signal),cp.real(cp.array(wavelet)),'valid')

    convolution = convolution.get()

    if verbose:
        signal = signal.get()
        wavelet = wavelet.get()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time, np.real(wavelet), color='tab:blue')
        ax.plot(time, np.imag(wavelet), color='tab:orange')

        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(convolution,color='tab:blue')
        plt.show()

    return convolution



#signal = np.sin(np.arange(0,100,0.001))
#wavelet_transform(signal, freq=1, discreteness=0.01,number_of_scycles=2,verbose = True)
#duration = timeit.timeit('wavelet_transform(signal, freq=1, discreteness=0.01,number_of_scycles=2,verbose = False)', number=1000, globals = globals())
#print(duration)
