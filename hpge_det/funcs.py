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
    d_MAF = [np.sum(d[i-l:i]) / l for i in range(l, LEN)]
    return np.concatenate((d[0:l], np.array(d_MAF)))


def apply_DIFF(d, l):
    d_DIFF = np.array([(d[i+l] - d[i-l]) for i in range(l, LEN-l-1)])
    return np.concatenate((d[0:l], np.array(d_DIFF), d[LEN-l-1:]))


def calc_en_int(d, i_start, i_stop, bl_shift):
    idx_max = np.argmax(d)
    int_start = idx_max - i_start
    int_stop = i_stop
    bl_start = idx_max - bl_shift
    bl_stop = bl_start + 128
    bl = np.mean(d[bl_start:bl_stop])
    en = np.sum(d[int_start:int_stop]) - (int_stop - int_start) * bl
    return en, int_start, int_stop, bl


def calc_en_trap(d, tau, l, k, bl_shift):
    #apply Moving average filter to signal (integration)
    d_MA = np.array([np.sum(d[i-30:i])/30 for i in range(30, LEN)])
    #differentiante signal
    d_diff = np.array([(d_MA[i+50] - d_MA[i-50]) / 100. for i in range(50, LEN-30-50)])
    cs, cTr = np.zeros(LEN), np.zeros(LEN)
    idx_max = np.argmax(d_diff[10:]) + 10 + 80
    bl_start = idx_max - bl_shift
    bl_stop = bl_start + 100
    bl = np.mean(d[bl_start:bl_stop])
    d_clr = d - bl
    d_clr[:bl_start] = 0.0
    dtau = np.exp(-1. / tau)
    g, m = dtau / (1 - dtau), 2 * (1 - dtau) / (1 + dtau) #see arxiv.org/pdf/1504.02039.pdf
    d_PZ = np.zeros(LEN)
    #d_PZ = d_clr * np.exp(np.arange(LEN) / tau)
    d_PZ[0] = m * (g * d_clr[0] + d_clr[0])
    for i in range(1, LEN):
        d_PZ[i] = d_PZ[i-1] - m * g * d_clr[i-1] + m * g * d_clr[i] + m * d_clr[i]#d_PZ[i] = m * (g * d_clr[i] + np.sum(d_clr[:i])) #should be :i+1

    #to kill lin decay of PZ corrected signal
    #!!!! MAY BE it will be better to PZ correct d_MA signal!!!
    #x_lin = np.arange(idx_max + 1500, idx_max + 2000)
    #y_lin = d_PZ[x_lin]
    #cp = np.polyfit(x_lin, y_lin, deg=1)
    #d_PZ -= cp[0] * np.arange(LEN)
        
    cTr = d_PZ
    for i in range(1, LEN):
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
    idxcs_max = int(idx_max + 1.5 * k) #was + 2 * k | for small l, k -> 3 * k
    en = np.mean(cs[idxcs_max:idxcs_max+(l)//2]) #was (l-k)//2 | was + 5000
            
    return en, cs, d_PZ, d_clr, idxcs_max, bl
    

def calc_CPpzc_RC1(dat, tau, RC, bl_shift):
    #apply Moving average filter to signal (integration)
    dat_MA = np.array([np.sum(dat[i-30:i])/30 for i in range(30, LEN)])
    #differentiate signal
    dat_diff = np.array([(dat_MA[i+50] - dat_MA[i-50]) / 100. for i in range(50, LEN-30-50)])
    idx_max = np.argmax(dat_diff[10:]) + 10
    bl_start = idx_max - bl_shift
    bl_stop = bl_start + 100
    bl = np.mean(dat[bl_start:bl_stop])

    dat_clr = dat - bl

    d, k = RC / (RC + 1), tau / RC
    A = (k * d + 1 - d) / (k + 1 - d)
    B = (k * d) / (k + 1 - RC)
    V0 = np.zeros(LEN)
    for i in range(2, LEN):
        V0[i] = A * (1 - d) * dat_clr[i] - B  * (1 - d) * dat_clr[i-1] + (d + B) * V0[i-1] - B * d * V0[i - 2]
    #en = np.sum(V0[idx_max:idx_max+5000])
    en = np.max(V0)
    
    return en, V0, dat_clr, idx_max


def calc_CR_RC1(dat, tau, RC, s_l, s_r, bl_shift):
    dat_MA = np.array([np.sum(dat[i-30:i])/30 for i in range(30, LEN)])
    #differentiate signal
    dat_diff = np.array([(dat_MA[i+50] - dat_MA[i-50]) / 100. for i in range(50, LEN-30-50)])
    idx_max = np.argmax(dat_diff[10:]) + 80 + 10
    bl_start = idx_max - bl_shift
    bl_stop = bl_start + 100
    bl = np.mean(dat[bl_start:bl_stop])

    dat_clr = dat - bl
    
    dtau = np.exp(-1. / tau)
    g, m = dtau / (1 - dtau), 2 * (1 - dtau) / (1 + dtau) #see arxiv.org/pdf/1504.02039.pdf
    d_PZ = np.zeros(LEN)
    #d_PZ = d_clr * np.exp(np.arange(LEN) / tau)
    d_PZ[0] = m * (g * dat_clr[0] + dat_clr[0])
    for i in range(1, LEN):
        d_PZ[i] = d_PZ[i-1] - m * g * dat_clr[i-1] + m * g * dat_clr[i] + m * dat_clr[i]#d_PZ[i] = m * (g * d_clr[i] + np.sum(d_clr[:i])) #should be :i+1

    y = np.zeros(LEN)
    a = 1.0 / RC
    alpha = np.exp(-1.0 / RC)
    for i in range(2, LEN):
        y[i] = 2 * alpha * y[i-1] - alpha**2 * y[i-2] + d_PZ[i] - alpha * (1.0 + a) * d_PZ[i-1]
    idx_ymax = np.argmax(y)
    en = np.sum(y[idx_ymax-s_l:idx_ymax+s_r]) / (s_l + s_r)

    return en, y, dat_clr, d_PZ, idx_ymax

def calc_CR_RC2(dat, tau, RC, s_l, s_r, bl_shift):
    dat_MA = np.array([np.sum(dat[i-30:i])/30 for i in range(30, LEN)])
    #differentiate signal
    dat_diff = np.array([(dat_MA[i+50] - dat_MA[i-50]) / 100. for i in range(50, LEN-30-50)])
    idx_max = np.argmax(dat_diff[10:]) + 80 + 10
    bl_start = idx_max - bl_shift
    bl_stop = bl_start + 100
    bl = np.mean(dat[bl_start:bl_stop])

    dat_clr = dat - bl
    
    dtau = np.exp(-1. / tau)
    g, m = dtau / (1 - dtau), 2 * (1 - dtau) / (1 + dtau) #see arxiv.org/pdf/1504.02039.pdf
    d_PZ = np.zeros(LEN)
    #d_PZ = d_clr * np.exp(np.arange(LEN) / tau)
    d_PZ[0] = m * (g * dat_clr[0] + dat_clr[0])
    for i in range(1, LEN):
        d_PZ[i] = d_PZ[i-1] - m * g * dat_clr[i-1] + m * g * dat_clr[i] + m * dat_clr[i]#d_PZ[i] = m * (g * d_clr[i] + np.sum(d_clr[:i])) #should be :i+1

    y = np.zeros(LEN)
    a = 1.0 / RC
    alpha = np.exp(-1.0 / RC)
    for i in range(2, LEN):
        y[i] = 3 * alpha * y[i-1] - 3 * alpha**2 * y[i-2] + alpha**3 * y[i-3] + alpha * (1 - a/2) * d_PZ[i-1] - alpha**2 * (1 + a/2) * d_PZ[i-2]

    idx_ymax = np.argmax(y)
    en = np.sum(y[idx_ymax-s_l:idx_ymax+s_r]) / (s_l + s_r)

    return en, y, dat_clr, d_PZ, idx_ymax


def calc_fwhm(a, x_l, x_r):
    arr = a[x_l:x_r]
    
    max_index = np.argmax(arr)
    #IF BAD PICK exit
    if arr[max_index] == 0:
        return (0, 0)

    print(max_index)
    more_than_max2 = np.where(arr > arr[max_index]/2)[0]
    print("more_than_max2 = ", more_than_max2)
        
    #find FWHM_left bound and FWHM_right bound
    yp_l = [more_than_max2[0]-1, more_than_max2[0]]
    xp_l = [arr[y] for y in yp_l]
    k_fwhm_l, b_fwhm_l = np.polyfit(xp_l, yp_l, 1)
    fwhm_l = k_fwhm_l * arr[max_index]/2 + b_fwhm_l
    yp_r = [more_than_max2[-1], more_than_max2[-1]+1]
    xp_r = [arr[y] for y in yp_r]
    k_fwhm_r, b_fwhm_r = np.polyfit(xp_r, yp_r, 1)
    fwhm_r = k_fwhm_r * arr[max_index]/2 + b_fwhm_r
    fwhm = fwhm_r-fwhm_l
    fwhm_y = arr[max_index]/2.0

    #set vars for plots
    fwhm_ch_l = k_fwhm_l * fwhm_y + b_fwhm_l#(fwhm_y / 2 - b_fwhm_l) / k_fwhm_l
    fwhm_ch_r = k_fwhm_r * fwhm_y + b_fwhm_r#(fwhm_y / 2 - b_fwhm_r) / k_fwhm_r
    print("fwhm_ch_l = {}, fwhm_ch_r = {}".format(x_l + fwhm_ch_l, x_r + fwhm_ch_r))
    print(yp_l, xp_l)
    print("k, b = {}, {}".format(k_fwhm_l, b_fwhm_l))
    print("by hand k, b = {}, {}".format((yp_l[0] - yp_l[1]) / (xp_l[0] - xp_l[1]),
                                         yp_l[0] - (yp_l[0] - yp_l[1]) / (xp_l[0] - xp_l[1]) * xp_l[0]))
    print(yp_r, xp_r)
    print("k, b = {}, {}".format(k_fwhm_r, b_fwhm_r))
    print("by hand k, b = {}, {}".format((yp_r[0] - yp_r[1]) / (xp_r[0] - xp_r[1]),
                                         yp_r[0] - (yp_r[0] - yp_r[1]) / (xp_r[0] - xp_r[1]) * xp_r[0]))
    
    #FWHM Error
    fwhm_err_l, fwhm_err_r = np.zeros(2), np.zeros(2)
    xp_l_err = [x+(abs(x))**0.5 for x in xp_l]
    k, b = np.polyfit(xp_l_err, yp_l, 1)
    fwhm_err_l[0] = k * arr[max_index]/2 + b
    xp_l_err = [x-(abs(x))**0.5 for x in xp_l]
    k, b = np.polyfit(xp_l_err, yp_l, 1)
    fwhm_err_l[1] = k * arr[max_index]/2 + b
    print(fwhm_l, fwhm_err_l[0], fwhm_err_l[1])

    xp_r_err = [x+(x)**0.5 for x in xp_r]
    k, b = np.polyfit(xp_r_err, yp_r, 1)
    fwhm_err_r[0] = k * arr[max_index]/2 + b
    xp_r_err = [x-(x)**0.5 for x in xp_r]
    k, b = np.polyfit(xp_r_err, yp_r, 1)
    fwhm_err_r[1] = k * arr[max_index]/2 + b
    print(fwhm_r, fwhm_err_r[0], fwhm_err_r[1])

    fwhm_err = ((fwhm_err_l[0]-fwhm_err_l[1])**2 + (fwhm_err_r[0]-fwhm_err_r[1])**2)**0.5

    return fwhm, fwhm_err
