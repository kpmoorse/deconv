import numpy as np
from scipy import interpolate
from scipy import signal
from scipy import special
import matplotlib.pyplot as plt

chi2 = lambda x,k: x**(float(k)/2-1)*np.exp(-x/2) / (2**(float(k)/2)*special.gamma(float(k)/2))
tt = np.arange(0, 50, 1)
tt = np.arange(0,1,0.001)
nk = 150

# Define signal
pulse = (tt > 1./3) * (tt < 2./3) - 0.5
multi = (tt%0.25 > 0.125) - 0.5
random = np.random.randint(0,2,len(tt)) - 0.5
pink = np.mod(np.cumsum(np.random.random(len(tt)) > 0.95), 2)

sig = pink

# Define convolution kernels
triangle = (-3*tt+2)*pulse
gauss = np.exp(-(tt-0.5)**2/(2*0.0025**2))
fluor = np.hstack((np.zeros(nk), chi2(tt*100,4)[:nk]))
integ = np.hstack((np.zeros(len(tt)), np.ones(len(tt))))
lpf = np.hstack((np.zeros(len(tt)), np.exp(-25*tt)))

kernel = fluor
kernel = np.array([1,2,1])

aconvolved = signal.convolve(sig, kernel)
adeconvolved = signal.deconvolve(aconvolved, kernel)[0]

plt.figure()
plt.plot(tt, sig, 'g')
plt.plot(tt, aconvolved[1:-1], 'r') 
plt.plot(tt, adeconvolved, 'b--')
plt.show()