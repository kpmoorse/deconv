import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

chi2 = lambda x,k: x**(float(k)/2-1)*np.exp(-x/2) / (2**(float(k)/2)*gamma(float(k)/2))
tt = np.arange(0,1,0.001)
n = len(tt)

# Define signals
m1 = np.mod(np.cumsum(np.random.random(len(tt)) > 0.95), 2) + np.random.normal(0,0.1,n)
m2 = np.mod(np.cumsum(np.random.random(len(tt)) > 0.95), 2) + np.random.normal(0,0.1,n)

# Define convolution kernels
fluor = np.hstack((np.zeros(len(tt)), chi2(tt*500,4)[:len(tt)]))
integ = np.hstack((np.zeros(len(tt)), np.ones(len(tt))))
lpf = np.hstack((np.zeros(len(tt)), np.exp(-25*tt)))

kernel = fluor
kernel = kernel / np.sum(kernel)

# Apply Convolution
conv1 = np.convolve(m1, kernel, mode='valid')
conv2 = np.convolve(m2, kernel, mode='valid')

# Plot results
plt.subplot(321)
plt.plot(tt, m1)

plt.subplot(323)
plt.plot(tt, m2)

plt.subplot(325)
plt.plot(kernel)

plt.subplot(122)
plt.plot(m1, m2, '.')
plt.plot(conv1, conv2, '.')

plt.show()