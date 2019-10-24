import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

chi2 = lambda x,k: x**(float(k)/2-1)*np.exp(-x/2) / (2**(float(k)/2)*gamma(float(k)/2))
tt = np.arange(0,1,0.001)

# Define signal
pulse = (tt > 1./3) * (tt < 2./3) - 0.5
multi = (tt%0.25 > 0.125) - 0.5
random = np.random.randint(0,2,len(tt)) - 0.5
pink = np.mod(np.cumsum(np.random.random(len(tt)) > 0.95), 2)

signal = pink

# Define convolution kernels
triangle = (-3*tt+2)*pulse
gauss = np.exp(-(tt-0.5)**2/(2*0.0025**2))
fluor = np.hstack((np.zeros(len(tt)), chi2(tt*100,4)[:len(tt)]))
integ = np.hstack((np.zeros(len(tt)), np.ones(len(tt))))
lpf = np.hstack((np.zeros(len(tt)), np.exp(-25*tt)))

kernel = fluor

# Apply Convolution
conv = np.convolve(signal, kernel, mode='valid')
print(conv)

# Plot results
plt.subplot(311)
plt.plot(tt, signal)
plt.subplot(312)
plt.plot(kernel)
plt.subplot(313)
plt.plot(tt, conv[:len(tt)])
plt.show()