import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss
import math as m

tt = np.arange(0,10*2e4,10)
n = len(tt)

# Define signals
m1 = np.mod(np.cumsum(np.random.random(len(tt)) > 1-1e-3), 2) + np.random.normal(0,0.1,n)
m2 = np.mod(np.cumsum(np.random.random(len(tt)) > 1-1e-3), 2) + np.random.normal(0,0.1,n)

# Import kernel
kernels = pd.read_csv('data/emp_kernels.csv')
tonic = kernels["i1_scaled"]
tonic = tonic[::10]
tonic = tonic / np.sum(tonic)

# Apply Convolution
conv1 = np.convolve(m1, tonic)
conv2 = np.convolve(m2, tonic)

# Apply Deconvolution
deconv1, _ = ss.deconvolve(conv1, tonic)
deconv2, _ = ss.deconvolve(conv2, tonic)

# Plot results
plt.subplot(321)
plt.plot(tt, m1)

plt.subplot(323)
plt.plot(tt, m2)

# plt.subplot(325)
# plt.plot(tonic)

# plt.subplot(122)
# plt.plot(m1, m2, '.')
# plt.plot(conv1, conv2, '.')

for scale in np.arange(0.5, 1.75, 0.25):

    stretch = ss.resample(tonic, int(m.floor(len(tonic)*scale)))
    stretch = stretch / np.sum(stretch)
    deconv1, _ = ss.deconvolve(conv1, stretch)
    deconv2, _ = ss.deconvolve(conv2, stretch)

    plt.subplot(325)
    plt.plot(stretch)

    plt.subplot(122)
    plt.plot(deconv1, deconv2, '.')
    plt.xlim([-0.5, 1.5]); plt.ylim([-0.5, 1.5])

plt.show()