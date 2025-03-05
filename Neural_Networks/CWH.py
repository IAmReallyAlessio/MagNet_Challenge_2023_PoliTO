from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

csv_B = 'N87\B_waveform[T].csv'
B_table = pd.read_csv(csv_B)

csv_f = 'N87/Frequency[Hz].csv'
f_list = pd.read_csv(csv_f)

csv_H = 'N87/H_waveform[Am-1].csv'
H_table = pd.read_csv(csv_H)

csv_W = 'N87/Volumetric_losses[Wm-3].csv'
W_list = pd.read_csv(csv_W)

n=1024 #samples
N=int(40616/4)

count=0
ind = 1

def plot_class(data, title, num_bins):
    bin_boundaries = np.linspace(min(data), max(data), num_bins + 1)
    hist, _ = np.histogram(data, bins=bin_boundaries)
    plt.bar(bin_boundaries[:-1], hist, width=(bin_boundaries[1]-bin_boundaries[0]), align='edge')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(title+': #'+str(ind))
    plt.grid()
    plt.show()

def power_i(l, alpha, beta, k, f):
    n = 1024

    der = []
    for key, value in l.items():
        for ii in range(len(value)):
            if ii < len(value) - 2:
                dv = (value[ii + 1] - value[ii]) * f * n
                der.append(dv)

    P_i_list = []
    for key, value in l.items():
        B_pkpk = float(max(value)) - float(min(value))
        P_B_i = k * ((abs(der) / B_pkpk) ** alpha) * (B_pkpk ** beta)
        t = (1 / f) * list(range(1, len(P_B_i)))
        P_i = integrate.trapz(P_B_i, t)
        P_i_list.append(P_i)


    P_B_i = k * ((abs(der) / B_pkpk) ** alpha) * (B_pkpk ** beta)
    t = (1/f)*list(range(1, 11))
    P_i = integrate.trapz(P_B_i, t)


def dict_lettura(l, num_classe, ind):
    x_values = np.arange(1024)

    for key, value in l.items():
        PK_PK = float(max(value)) - float(min(value))
        plt.plot(x_values, value/PK_PK)

    plt.xlabel('Single Period')
    plt.ylabel('Normalized Induction over Pk2Pk')
    plt.title('Form Factor Type: # %1d.%1d, %3d' % (ind, num_classe,len(l)))
    # ,str(ind),str(num_classe)
    plt.grid(True)
    #plt.hold(True)
    plt.show()

ind=1
while ind <= 4:
    fac_forma = []
    fac_cresta = []
    l1 = dict()
    l2 = dict()
    l3 = dict()
    l4 = dict()
    l5 = dict()
    l6 = dict()
    l7 = dict()
    l8 = dict()

    x_values = []
    y_values = []
    der_forma=[]
    for n_ in range((ind-1)*(N-1),ind*(N-1)):
        count=count+1
        row = B_table.iloc[n_] # estraggo i-esima riga da tabella induzione

        f = f_list.iloc[n_, 0] # estraggo i-esima riga da tabella frequenza

        # RMS
        row_rms = []
        for elem in row:
            row_rms.append(float(elem) ** 2) # inserisco dati alla seconda

        t = []
        freq_meas = float(f)
        for i in range(1024):
            t.append(i * 1 / (freq_meas * 1024)) # calcolo del tempo

        # RMS
        Integ_RMS = integrate.trapz(row_rms, t)
        RMS = (freq_meas * Integ_RMS) ** 0.5

        B = []
        B_abs= []
        for thing in row:
            B.append(float(thing)) # inutile
            B_abs.append(abs(float(thing)))

        # Peak-Peak
        P_P = float(max(row)) - float(min(row))

        # Valore Medio Raddrizzato
        V_rad = 0
        for elem in B_abs:
            V_rad = V_rad+elem
        V_m_rad = V_rad/1024

        forma_=RMS/V_m_rad

        if forma_ < 1.04:
            l1[n_]=row
        elif 1.06 < forma_ < 1.075:
            l2[n_]=row
        elif 1.08 < forma_ < 1.085:
            l3[n_]=row
        elif 1.10 < forma_ < 1.108:
            l4[n_]=row
        elif 1.108 <= forma_ < 1.12:
            # derivata prima
            der = []
            for ii in range(len(row)):
                if ii<len(row)-2:
                    dv = (row[ii+1]-row[ii]) * freq_meas * n
                    der.append(dv)
            # derivata seconda
            der2 = []
            for ii in range(len(der)):
                if ii < len(der) - 2:
                    dv = (der[ii + 1] - der[ii]) * freq_meas * n
                    der2.append(dv)

            k=15*(3.142*freq_meas)**2
            if max(der2) < k*max(row):
                l5[n_] = row
            else:
                l6[n_] = row

        elif 1.125 < forma_ < 1.14:
            l7[n_] = row
        elif forma_ > 1.14:
            l8[n_]=row
        else:
            print('errore')

    dict_lettura(l1, 1, ind)
    dict_lettura(l2, 2, ind)
    dict_lettura(l3, 3, ind)
    dict_lettura(l4, 4, ind)
    dict_lettura(l5, 5, ind)
    dict_lettura(l6, 6, ind)
    dict_lettura(l7, 7, ind)
    dict_lettura(l8, 8, ind)

    ind+=1
