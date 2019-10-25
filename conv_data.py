import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as ss

tt = np.arange(0,10*2e4,10)
n = len(tt)

# Define signals
m1 = np.mod(np.cumsum(np.random.random(len(tt)) > 1-1e-3), 2) + np.random.normal(0,0.1,n)
m2 = np.mod(np.cumsum(np.random.random(len(tt)) > 1-1e-3), 2) + np.random.normal(0,0.1,n)

# Import kernel
kernels = pd.read_csv('data/emp_kernels.csv')
tonic = kernels["i1_scaled"]
tonic = np.hstack((np.zeros(len(tonic)), tonic))
tonic = tonic[::10]
tonic = tonic / np.sum(tonic)

# Apply Convolution
conv1 = np.convolve(m1, tonic, mode='valid')
conv2 = np.convolve(m2, tonic, mode='valid')

# Apply Deconvolution
# deconv1, _ = ss.deconvolve(conv1, tonic)
# deconv2, _ = ss.deconvolve(conv2, tonic)

# Plot results
plt.subplot(321)
plt.plot(tt, m1)

plt.subplot(323)
plt.plot(tt, m2)

plt.subplot(325)
plt.plot(tonic)

plt.subplot(122)
plt.plot(m1, m2, '.')
plt.plot(conv1, conv2, '.')
# plt.plot(deconv1, deconv2, '.')

plt.show()