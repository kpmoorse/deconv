import numpy as np
import matplotlib.pyplot as plt

# Return the smallest power of 2 greater than or equal to a number
def expceil(x, a=2):

    temp = np.log(x)/np.log(a)
    return a**np.ceil(temp)

# Take the fourier transform of the windowed input function
# Return amplitude, phase, and frequency-spacing
def fft(ft, t, pad=1, window=np.hanning):

    # Extract sample period
    if len(t) > 0:
        dt = np.diff(t[:2])[0]
        assert np.all(np.diff(t) - dt < dt/1e6)
    else:
        dt = t

    if window:
        ft = window(len(ft))*ft

    # Find power-of-two pad length and apply transform
    N = int(expceil(len(ft)*pad))
    ff = np.fft.fft(ft, N)
    ff = ff[:N/2]
    f = np.fft.fftfreq(N, dt)[:N/2]

    # Separate amplitude and phase
    amp = np.abs(ff)
    print(np.sum(amp**2))
    ph = np.angle(ff)

    return (amp, ph, f)

def ifft(ff, f):

    pass

if __name__ == '__main__':

    gauss = lambda x, x0, s: np.exp(-(x-x0)**2/(2*s**2))
    t = np.arange(0, 10, 0.1)
    ft = 2*np.sin(2*2*np.pi*t) + np.sin(6*2*np.pi*t)
    ft = np.sin(2*np.pi*t)
    # ft = gauss(t, 5, 2)
    (amp, ph, f) = fft(ft, t, pad=2)

    plt.subplot(211)
    plt.plot(t, ft)
    plt.subplot(212)
    plt.plot(f, amp)
    plt.show()