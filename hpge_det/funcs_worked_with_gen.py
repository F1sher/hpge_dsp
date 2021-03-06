import numpy as np
LEN = 32768

def calc_en_max(d, bl_shift): 
    idx_max = np.argmax(d)
    bl_start = idx_max - bl_shift #bl_shift \approx 300
    bl_stop = bl_start + 128
    bl = np.mean(d[bl_start:bl_stop])
    en = d[idx_max] - bl
    return en, idx_max, d[idx_max], bl


def apply_MAF(d, l):
    d_MAF = [np.sum(d[i-l:i]) for i in range(l, LEN_EVENT)]
    return np.concatenate((d[0:l], np.array(d_MAF)))


def calc_en_int(d, i_start, i_stop, bl_shift):
    idx_max = np.argmax(d)
    int_start = idx_max - i_start
    int_stop = i_stop
    bl_start = idx_max - bl_shift
    bl_stop = bl_start + 128
    bl = np.mean(d[bl_start:bl_stop])
    en = np.sum(d[int_start:int_stop]) - (int_stop - int_start) * bl
    return en, int_start, int_stop, bl


def calc_en_trap(d, tau, l, k, bl_shift, le=LEN):
    #apply Moving average filter to signal (integration)
    d_MA = np.array([np.sum(d[i-30:i])/30 for i in range(30, le)])
    #differentiante signal
    d_diff = np.array([(d_MA[i+50] - d_MA[i-50]) / 100. for i in range(50, le-30-50)])
    cs, cTr = np.zeros(le), np.zeros(le)
    idx_max = np.argmax(d_diff)
    bl_start = idx_max - bl_shift
    bl_stop = bl_start + 50
    bl = np.mean(d[bl_start:bl_stop])
    d_clr = d - bl
    d_clr[:bl_start] = 0.0
    dtau = np.exp(-1. / tau)
    g, m = dtau / (1 - dtau), 2 * (1 - dtau) / (1 + dtau) #see arxiv.org/pdf/1504.02039.pdf
    d_PZ = np.zeros(le)
    #d_PZ = d_clr * np.exp(np.arange(l) / tau)
    d_PZ[0] = m * (g * d_clr[0] + d_clr[0])
    for i in range(1, le):
        d_PZ[i] = d_PZ[i-1] - m * g * d_clr[i-1] + m * g * d_clr[i] + m * d_clr[i]#d_PZ[i] = m * (g * d_clr[i] + np.sum(d_clr[:i])) #should be :i+1
    cTr = d_PZ
    for i in range(1, le):
        if (i - l - k >= 0):
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k] - cTr[i-l] + cTr[i-k-l]
        elif (i - l >= 0):
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k] - cTr[i-l]
        elif (i - k >= 0):
            cs[i] = cs[i-1] + cTr[i] - cTr[i-k]
        else:
            cs[i] = cs[i-1] + cTr[i]
    #for generator
    #idxcs_max = np.argmax(cs)
    #for detector
    idxcs_max = int(idx_max + 1.5 * k)#for small l, k - 3 * k
    en = np.mean(cs[idxcs_max:idxcs_max+(l-k)//3])
            
    return en, cs, d_PZ, d_clr, idxcs_max, bl
    
