#!/usr/bin/env python

#TODO
#1) working with binary data files
#2) check for wave1 data
#3) need better algorithm to clear raw data from noise or collect longer signals

#What I changed for HPGe
#1) Make signal negative (see main())
#2) Add exception checker in calc_RT()
#3) added import exc_info()
#4) Some scaling of CR_RC


from itertools import islice 
from sys import exc_info
import numpy as np
import matplotlib
matplotlib.use("GTK3Agg", warn=False, force=True)
import matplotlib.pyplot as plt
from scipy import interpolate


LEN_EVENT = 32768 #on the Ge detector will be 10k and 16384

def load_data(filepath, start=0, stop=100, format="ascii"):
    if format == "ascii":
        with open(filepath, "r") as fd:
            data = np.loadtxt(islice(fd, start*LEN_EVENT, stop*LEN_EVENT), dtype=int)#[start*LEN_EVENT:stop*LEN_EVENT]
    elif format == "bin":
        data = np.array([])

    return data.reshape((-1, LEN_EVENT))


def clear_raw_data(data):
    DEL_THRESHOLD = 1700
    idxs_remove = []
    for i, d in enumerate(data):
        d_sorted = np.sort(d)
        min_avg = np.sum(d_sorted[0:20]) / 20.0
        max_avg = np.sum(d_sorted[-20:]) / 20.0
        
        #if (max(d) - min(d)) < DEL_THRESHOLD:
        if np.abs(max_avg - min_avg) < DEL_THRESHOLD:
            idxs_remove.append(i)
        if i < 10:
            print("max - min = {:d}".format(max(d) - min(d)))
            print(idxs_remove)

    #print(idxs_remove)
            
    return np.delete(data, idxs_remove, axis=0)


def correct_bl(d, bl_start=10, bl_stop=50):
    bl = np.sum(d[bl_start:bl_stop]) / (bl_stop - bl_start)
    d = d - bl
    d[d > 0] = 0
    return d


def fit_exp(data, n_start=0, n_stop=10):
    taus, As = [], []
    for d in data[n_start:n_stop]:
        dc = correct_bl(d)
        idx_min = np.where(dc == min(dc))[0][0]
        idx_min = 1250
        print(np.where(dc == min(dc)))
        print("idx_min = {}".format(idx_min))
        #y = A exp(tau*x)
        #ln(y) = ln(A) + tau*x
        p = np.polyfit(np.arange(LEN_EVENT)[idx_min:LEN_EVENT], np.log(np.abs(dc[idx_min:LEN_EVENT]) + 1), 1)
        tau, A = p[0], np.exp(p[1])
        print("tau = {:.2e}, A = {:.2e}".format(tau, A))
        taus.append(tau)
        As.append(A)

    return sum(taus) / (n_stop - n_start), sum(As) / (n_stop - n_start)


def PZ_corr(dat, tau):
    y = np.zeros(LEN_EVENT)
    for i in range(1, LEN_EVENT):
        y[i] = y[i-1] + dat[i] - dat[i-1] * (1.0 + tau)
    """
    d = np.exp(1.0 / tau)
    g = d / (1.0 - d)
    m = 2.0 * (1.0 - d) / (1.0 + d)
    #b1 = np.exp(-tau)
    #a0 = (1.0 + b1) / 2.0
    #a1 = -(1.0 + b1) / 2.0
    y = np.zeros(LEN_EVENT)
    for i in range(1, LEN_EVENT-1):
        y[i] = (g * dat[i] + np.sum(dat[0:i])) * m #b1 * y[i-1] + a0 * d[i] + a1 * d[i-1] + d[i-1]
    """
    return y


def make_trap_loops(d, tau, l, k):
    #!!!! use correct_bl() !!!!
    bl_start, bl_stop = 0, 10
    bl = np.sum(d[bl_start:bl_stop]) / (bl_stop - bl_start)
    d_clear = np.zeros(LEN_EVENT)
    cTr = np.zeros(LEN_EVENT)
    cs = np.zeros(LEN_EVENT)
    
    for i in range(LEN_EVENT):
        if (d[i] - bl > 0):
            d_clear[i] = 0.0
        else:
            d_clear[i] = d[i] - bl
        cTr[i], cs[i] = d_clear[i], d_clear[i]

    cTr = PZ_corr(d, tau)
    #for i in range(1, LEN_EVENT):
    #    cTr = d_clear[i] * np.exp(-tau * i)

    for i in range(1, LEN_EVENT):
        if (i - l - k >= 0):
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k] - cTr[i-l] + cTr[i-k-l]
        elif (i - l >= 0):
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k] - cTr[i-l]
        elif (i - k >= 0):
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k]
        else:
            cs[i] = cs[i-1] + cTr[i]

    return d_clear, cTr, cs / 1.0e3


def calc_trap_s(d, cs, l, k):
    AVG_TRAP = (l - k - 1) // 2
    AVG_TRAP = 800
    #I_MIN_S_SHIFT_TRAP = 20
    #for i in range(k, LEN_EVENT):
    #    if (d[i] <= 0.995 * d[i-1]):
    #        i_min_s = i + k + I_MIN_S_SHIFT_TRAP
    #        break
    i_min_s = np.where(d == min(d))[0][0] + k
    i_min_s = 200 + k
    #print("i_min_s = ", i_min_s, type(i_min_s))
        
    if (i_min_s + l - k > LEN_EVENT - 1):
        return 0.0
    res = np.sum(cs[i_min_s:i_min_s + AVG_TRAP]) / AVG_TRAP
    
    return res


def calc_CR_RC(d, tau, RC):
    bl_start, bl_stop = 0, 10
    bl = np.sum(d[bl_start:bl_stop]) / (bl_stop - bl_start)
    y = np.zeros(LEN_EVENT)
    d_clear = np.zeros(LEN_EVENT)
    for i in range(LEN_EVENT):
        if (d[i] - bl > 0):
            d_clear[i] = 0.0
        else:
            d_clear[i] = d[i] - bl

    a  = 1.0 / RC
    alpha = np.exp(-a)

    for i in range(2, LEN_EVENT):
        y[i] = 2 * alpha * y[i-1] - alpha**2 * y[i - 2] + d_clear[i] - alpha * (1 + a) * d_clear[i - 1]

    return y / 1.0e3

def calc_CR_RC_s(y, start=50, stop=5000, c_norm=1.0):
    return np.min(y[start:stop])
#res = np.sum(y[start:stop]) / c_norm
#    return res / (stop - start)
    

def calc_RT(d):
    bl_start, bl_stop = 0, 10
    bl = np.sum(d[bl_start:bl_stop]) / (bl_stop - bl_start)
    d_clear = np.zeros(LEN_EVENT)
    for i in range(LEN_EVENT):
        if (d[i] - bl > 0):
            d_clear[i] = 0.0
        else:
            d_clear[i] = d[i] - bl

    try:
        ampl_d = min(d_clear)
        i_start = np.where(d_clear < 0.1 * ampl_d)[0][0]
        p1_start = np.polyfit(d_clear[i_start-1:i_start+1], [i_start-1, i_start], 1)
        RT_start = np.poly1d(p1_start)(0.1 * ampl_d)

        i_stop = np.where(d_clear < 0.9 * ampl_d)[0][0]
        p1_stop = np.polyfit(d_clear[i_stop-1:i_stop+1], [i_stop-1, i_stop], 1)
        RT_stop = np.poly1d(p1_stop)(0.9 * ampl_d)
        #print("i_start = {}, stop = {}".format(i_start, i_stop))
        #print("RT start = {}, stop = {}".format(RT_start, RT_stop))
    except:
        print("Error in calc_RT(): ", exc_info()[0])
        return -1.0
    
    return (RT_stop - RT_start)


def calc_PSD(d, short_g=3000, long_g=10000):
    bl_start, bl_stop = 0, 10
    bl = np.sum(d[bl_start:bl_stop]) / (bl_stop - bl_start)
    d_clear = np.zeros(LEN_EVENT)
    for i in range(LEN_EVENT):
        if (d[i] - bl > 0):
            d_clear[i] = 0.0
        else:
            d_clear[i] = d[i] - bl

    short_s = np.sum(d[:short_g])
    long_s = np.sum(d[:long_g])

    return (long_s - short_s) / long_s


def calc_histo_en(data, alg="integral", tau=0.6):
    en = np.zeros(data.shape[0], dtype=float)

    if alg == "integral":
        s_start, s_stop = 500, LEN_EVENT
        bg_start, bg_stop = 32, 128
        for i, d in enumerate(data):
            norm_k = 1.0e4
            bg = np.sum(d[bg_start:bg_stop]) / (bg_stop - bg_start)
            s = (bg - d) / norm_k
            en[i] = np.abs(np.sum(s[s_start:s_stop]))
            np.savetxt("./histo_EN_integral.txt", en, fmt="%.2f")
    elif alg == "trap":
        l, k = 4000, 3000
        for i, d in enumerate(data):
            _, _, cs = make_trap_loops(d, tau, l, k)
            trap_s = calc_trap_s(d, cs, l, k)
            en[i] = np.abs(trap_s)
            print("en[{:d}] = {:.1f}".format(i, en[i]))
            np.savetxt("./histo_EN_trap.txt", en, fmt="%.2f")
    elif alg == "CR_RC":
        RC = 4000
        for i, d in enumerate(data):
            y_CR_RC = calc_CR_RC(d, tau, RC) 
            CR_RC_s = calc_CR_RC_s(y_CR_RC,  start=0, stop=LEN_EVENT)
            en[i] = np.abs(CR_RC_s)
            np.savetxt("./histo_EN_CR_RC.txt", en, fmt="%.2f")
    elif alg == "max":
        bg_start, bg_stop = 32, 128
        for i, d in enumerate(data):
            bg = np.sum(d[bg_start:bg_stop]) / (bg_stop - bg_start)
            en[i] = np.abs(min(d) - bg)

    en_avg, en_std = np.mean(en), np.std(en)
    print("EN: {:.4f} +- {:.4f}".format(en_avg, en_std))
    
    return np.histogram(en, bins=np.arange(0, 4095, 1))


def calc_histo_RT(data):
    RTs = np.zeros(data.shape[0], dtype=float)
    for i, d in enumerate(data):
        RT = calc_RT(d)
        #print("i = {} RT = {}".format(i, RT))
        RTs[i] = RT

    rt_avg, rt_std = np.mean(RTs), np.std(RTs)   
    print("RT: {} +- {}".format(rt_avg, rt_std))
        
    return np.histogram(RTs, bins=np.arange(0, 100, 0.1))


def calc_histo_PSD(data):
    PSDs = np.zeros(data.shape[0], dtype=float)
    for i, d in enumerate(data):
        PSD = calc_PSD(d)
        print("i = {} PSD = {:.6f}".format(i, PSD))
        PSDs[i] = PSD

    PSD_avg, PSD_std = np.mean(PSDs), np.std(PSDs)
    print("PSD: {} +- {}".format(PSD_avg, PSD_std))
        
    return np.histogram(PSDs, bins=np.arange(0, 1, 1e-6))


def main():
    data_filepath = "../data/3exp_wave1.txt"
    data_w1 = -load_data(data_filepath, start=0, stop=2000) #-load_data for raw data from HPGe det

    """
    ###To clear data
    dataclr_filepath = data_filepath[:-4] + "_clr" + data_filepath[-4:]
    data_w1 = clear_raw_data(data_w1)
    np.savetxt(dataclr_filepath,
               data_w1,
               fmt="%d",
               delimiter="\n")
    
    #en_w1 = calc_histo_en(data_w1)
    #np.savetxt("./en_w1_histo.txt", en_w1[0])
    #plt.plot(en_w1[0])
    #plt.show()
    """
    
    #histo_en_int = calc_histo_en(data_w1, alg="integral")
    histo_en_max = calc_histo_en(data_w1, alg="max")
    #try:
    #    np.savetxt("./histo_EN_integral_.txt", (histo_en_int[0], histo_en_int[1]), delimiter=" ")
    #except:
    #    None
    #histo_en_trap = calc_histo_en(data_w1, alg="trap", tau=-9.0e-06)
    #histo_en_CR_RC = calc_histo_en(data_w1, alg="CR_RC", tau=-2.0e-05)#tau=-1.6779740e-05
    #print("sums Int: {:.1f}, Trap: {:.1f}".format(np.sum(histo_en_trap[0])))
    #histo_RT = calc_histo_RT(data_w1)
    #histo_PSD = calc_histo_PSD(data_w1)
    #plt.plot(histo_en_int[1][:-1], histo_en_int[0], ".", label="Histo integral en")
    plt.plot(histo_en_max[1][:-1], histo_en_max[0], ".", label="Histo max en")
    #plt.plot(histo_en_trap[1][:-1], histo_en_trap[0], ".", label="Histo trap en")
    #plt.plot(histo_en_CR_RC[1][:-1], histo_en_CR_RC[0], ".", label="Histo CR-RC en")
    #plt.plot(histo_RT[1][:-1], histo_RT[0], label="Histo RT")
    #plt.plot(histo_PSD[1][:-1], histo_PSD[0], label="Histo PSD")
    plt.legend(loc="upper right")
    plt.show()
    
    
    ###TO PLOT/CHECK FILTERS
    """
    tau, A = fit_exp(data_w1, n_stop=1)
    print("tau = {}\nA = {}".format(tau, A))
    tau = 9e-06
    plt.plot(data_w1[0] * np.exp(tau * np.arange(LEN_EVENT)), label="exp decay removed")
    plt.plot(data_w1[0])
    #plt.plot(np.exp(-tau * np.arange(LEN_EVENT)))
    plt.show()
    
    for i in [0, 1, 5]:
        tau = -9.0e-06#-1.1672976805488264e-05
        l, k = 4000, 3000
        d_clear, cTr, cs = make_trap_loops(data_w1[i], tau, l, k)
        trap_s = calc_trap_s(data_w1[i], cs, l, k)
        print("trap_s = {:.2f}".format(trap_s))
        
        #RC = 4000
        #y_CR_RC = calc_CR_RC(data_w1[i], tau, RC) 
        #CR_RC_s = calc_CR_RC_s(y_CR_RC, start=50, stop=LEN_EVENT)
        #print("CR_RC_s = {:.2f}".format(CR_RC_s))

        #RT = calc_RT(data_w1[i])
        #print("RT = {:.2f}".format(RT))
        
        plt.plot(data_w1[i], label="raw data")
        dbl_corr = correct_bl(data_w1[i])
        plt.plot(dbl_corr, label="bl_corr")
        plt.plot(d_clear, label="d_clear")
        plt.plot(cTr, label="cTr")
        plt.plot(cs, label="cs")
        #plt.plot(y_CR_RC, label="CR_RC")
        plt.plot(PZ_corr(data_w1[i], tau), label="PZ corrected")
        plt.legend(loc="upper right")
        plt.show()
    """

if __name__ == "__main__":
    main()
