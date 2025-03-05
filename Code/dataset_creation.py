import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import scipy as sc
import json

basepath = Path(__file__).parent
# print(basepath)
foldername = 'N87_preset'
filenames = os.listdir(basepath / foldername)
db = pd.DataFrame()
filepath = basepath / foldername / filenames[0]
# print(filepath)
B_waveform = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)
filepath = basepath / foldername / filenames[1]
Frequency = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)
filepath = basepath / foldername / filenames[2]
H_waveform = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)
filepath = basepath / foldername / filenames[3]
Temperature = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)
filepath = basepath / foldername / filenames[4]
Power_losses = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0)

aux = B_waveform.to_numpy()
Timestep = 1/(len(aux[0, :])*Frequency.to_numpy())
aux_dB = np.diff(aux)/Timestep

B_pkpk = aux.ptp(axis=1)
B_rms = np.sqrt(np.mean(aux**2, axis=1))
B_arv = np.mean(abs(aux), axis=1)
k_f_B = B_rms/B_arv

dB_pkpk = aux_dB.ptp(axis=1)
dB_rms = np.sqrt(np.mean(aux_dB**2, axis=1))
dB_arv = np.mean(abs(aux_dB), axis=1)
k_f_dB = dB_rms/dB_arv

aux_df = np.stack((B_pkpk, B_rms, B_arv, k_f_B, dB_pkpk, dB_rms, dB_arv, k_f_dB), axis=1)
df = pd.DataFrame(aux_df, columns=['B_pkpk', 'B_rms', 'B_arv', 'k_f_B', 'dB_pkpk', 'dB_rms', 'dB_arv', 'k_f_dB'])
df['Frequency'] = Frequency.values
df['Power_Loss'] = Power_losses.values
df['Temperature'] = Temperature.values
# df_sin = df.loc[(df['k_f_B'] > np.pi/(2*2**0.5)*0.9985) & (df['k_f_B'] < np.pi/(2*2**0.5)*1.0015)]
df_tri = df.loc[(df['k_f_B'] > 2/(3**0.5)*0.9995) & (df['k_f_B'] < 2/(3**0.5)*1.0015)]
# print(np.shape(df_sin))
# print(df_sin)
# print(df_sin['Frequency'])
# print(df_sin['Power_Loss'])

# sin_values = []
# for x in aux:
#     B_rms = np.sqrt(np.mean(x ** 2, axis=0))
#     B_arv = np.mean(abs(x), axis=0)
#     k_f_B = B_rms / B_arv
#     if k_f_B > np.pi/(2*2**0.5)*0.9985 and k_f_B < np.pi/(2*2**0.5)*1.0015:
#         sin_values.append(x)
# sin_values = np.array(sin_values)
tri_values = []
for x in aux:
    B_rms = np.sqrt(np.mean(x ** 2, axis=0))
    B_arv = np.mean(abs(x), axis=0)
    k_f_B = B_rms / B_arv
    if k_f_B > 2/(3**0.5)*0.9995 and k_f_B < 2/(3**0.5)*1.0015:
        tri_values.append(x)
tri_values = np.array(tri_values)

# print(np.shape(sin_values))
# for y in sin_values:
#     plt.plot(y)
#     plt.show()

rms = np.sqrt(np.mean(tri_values**2, axis=1))
duty_ratios = []
for y in tri_values:
    x = np.linspace(0, 1024, 1024)
    # plt.plot(y)
    # plt.show()
    f = sc.interpolate.BSpline(x, y, 2)
    ynew = f(x)
    # plt.plot(x, y, 'o', x, ynew, '-')
    # plt.show()
    delta_f = f.derivative(nu=1)
    y_delta = delta_f(x)
    # plt.plot(x, y_delta, '-')
    # plt.show()
    n_pos = 0
    n_neg = 0
    for j in y_delta:
        if j > 0:
            n_pos += 1
        else:
            n_neg += 1
    duty_ratio = n_pos/(n_pos+n_neg)
    duty_ratios.append(duty_ratio)
duty_ratios = np.array(duty_ratios)
# print(duty_ratios)

# print(np.shape(duty_ratios))
# print(np.shape(rms))
# print(np.shape(df_tri['Frequency']))
# print(np.shape(df_tri['Temperature']))
# print(np.shape(df_tri['Power_Loss']))

with open("Dataset_tri_N87_preset.json", 'w') as file:
    data = {}
    data['Flux_Density'] = rms.tolist()
    data['Frequency'] = df_tri['Frequency'].to_numpy().tolist()
    data['Temperature'] = df_tri['Temperature'].to_numpy().tolist()
    # data['B_pkpk'] = df_sin['B_pkpk'].tolist()
    data['Duty_Ratio'] = duty_ratios.tolist()
    data['Power_Loss'] = df_tri['Power_Loss'].to_numpy().tolist()
    json_data = json.dumps(data)
    file.write(json_data)
    # print(json_data)