import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import scipy as sc
import json
from sklearn.preprocessing import StandardScaler

from Aux_fun import db_read_trapz

basepath = Path(__file__).parent
# print(basepath)
foldername = 'N87'
filenames = os.listdir(basepath / foldername)
db = pd.DataFrame()

filepath = basepath / foldername / filenames[0]
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
fdc = B_pkpk / 2 / B_rms

aux_df = np.stack((B_pkpk, B_rms, B_arv, k_f_B, fdc), axis=1)
df = pd.DataFrame(aux_df, columns=['B_pkpk', 'B_rms', 'B_arv', 'k_f_B', 'fdc'])
df['Frequency'] = Frequency.values
df['Power_Loss'] = Power_losses.values
df['Temperature'] = Temperature.values

condizione_tri = (np.isclose(df['k_f_B'], 2 / (3 ** 0.5), rtol=0.05)) & (np.isclose(df['fdc'], 3 ** 0.5, rtol=0.05))
condizione_sin = (np.isclose(df['k_f_B'], np.pi / (2 * 2 ** 0.5), rtol=0.006)) & (np.isclose(df['fdc'], 2 ** 0.5, rtol=0.006))

indici = set(df.index)
indici_righe_tri = df.index[condizione_tri].tolist()
indici_righe_sin = df.index[condizione_sin].tolist()
indici_righe_trapz = indici - set(indici_righe_tri) - set(indici_righe_sin)
indici_righe_trapz = list(indici_righe_trapz)

mat_type = 'N87_val'
aux_trapz, wave_trapz_old = db_read_trapz(mat_type)

df_trapz = df.loc[indici_righe_trapz]
B_pkpk = aux_trapz['pk_pk'] * 1e-3

T = 1 / Frequency
aux1 = np.arange(0, 1023, 8)
aux2 = np.linspace(0, 1023, 1024)
wave_trapz = np.interp(aux1,aux2,wave_trapz_old)

dt = T / len(wave_trapz)
der_B = np.diff(wave_trapz) / dt
f_eq_orig = 0.5 * np.abs(der_B) / B_pkpk

df['Frequency_eq'] = f_eq_orig
f_eq = list(filter(lambda x: x < 5e3, f_eq_orig))
select_indices = list(np.where(df['Frequency_eq'].values == f_eq)[0])
df_trapz = df_trapz.iloc[select_indices]


sc = StandardScaler()
rms = np.array(df_trapz['B_pkpk']).reshape(-1, 1)
sc.fit(rms)
rms_norm = sc.transform(rms)
f = df_trapz['Frequency'].to_numpy().reshape(-1, 1)
sc.fit(f)
f_norm = sc.transform(f)
temp = df_trapz['Temperature'].to_numpy().reshape(-1, 1)
sc.fit(temp)
temp_norm = sc.transform(temp)
power = df_trapz['Power_Loss'].to_numpy().reshape(-1, 1)
sc.fit(power)
power_norm = sc.transform(power)

with open("Dataset_tri_N87_norm.json", 'w') as file:
    data = {}
    data['Flux_Density'] = rms_norm.flatten().tolist()
    data['Frequency'] = f_norm.flatten().tolist()
    data['Temperature'] = temp_norm.flatten().tolist()
    data['Power_Loss'] = power_norm.flatten().tolist()
    json_data = json.dumps(data)
    file.write(json_data)
  