#!/usr/bin/env python

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def g(t, sigma=1.0, delta=0.0):
    return np.exp(-(t - delta)**2 / (2.0 * sigma**2)) / ((2.0 * np.pi)**0.5 * sigma)


def sgnl(t, t_r, t_d):
    return (np.exp(-t/t_d) - np.exp(-t/t_r)) / (t_d - t_r)


if __name__ == "__main__":
    tt = np.arange(0, 100, 0.01)
    sigma, delta = 0.5, 10.0
    t_r, t_d = 5.0, 50.0
    model_sgnl = signal.convolve(sgnl(tt, t_r, t_d),
                                   g(tt, sigma, delta),
                                   mode="same") / sum(g(tt))
    t_r2, t_d2 = 3.0, 10.0
    model_sgnl2 = signal.convolve(sgnl(tt, t_r2, t_d2),
                                   g(tt, sigma, delta),
                                   mode="same") / sum(g(tt))
    
    print("S(sgnl, {}, {}) = {:e}\tS(model) = {:e}\nS(sgnl, {}, {}) = {:e}\tS(model) = {:e}".
          format(t_r, t_d, np.sum(sgnl(tt, t_r, t_d)), np.sum(model_sgnl),
                 t_r2, t_d2, np.sum(sgnl(tt, t_r2, t_d2)), np.sum(model_sgnl2)))
                 
    plt.plot(sgnl(tt, t_r, t_d), label="1st sgnl")
    plt.plot(sgnl(tt, t_r2, t_d2), label="2nd sgnl")
    plt.plot(model_sgnl, label="Model sgnl")
    plt.plot(model_sgnl2, label="Model sgnl2")
    plt.legend(loc="upper right")
    plt.show()
    
    #plt.plot(g(tt, sigma, delta))
    #plt.plot(sgnl(tt, t_r, t_d))
    #plt.plot(model_sgnl)
    #plt.show()

    
    """
    fig, (ax_g, ax_signal, ax_filt) = plt.subplots(3, 1, sharex=True)
    ax_g.plot(g(tt))
    ax_g.set_title("Gauss IRF")
    ax_signal.plot(sgnl(tt, t_r, t_d))
    ax_signal.set_title("Signal")
    fig.tight_layout()
    fig.show()
    """
