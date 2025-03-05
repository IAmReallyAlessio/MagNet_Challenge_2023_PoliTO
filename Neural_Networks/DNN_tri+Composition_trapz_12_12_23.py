import csv
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from Aux_fun import db_read_trapz

mat_type = 'N87_val'

aux_trapz, wave_trapz_old = db_read_trapz(mat_type)
current_file_path = os.path.abspath(__file__)
basepath = os.path.dirname(current_file_path)
DNN_model_path = os.path.join(basepath, 'Model', 'DNN_tri_N87.keras')
DNN_mean_path = os.path.join(basepath, 'Model', 'Mean&Sd_tri_N87.csv')

model = keras.models.load_model(DNN_model_path)
# model.summary()

mean_input = np.empty(4, dtype=float)
std_input = np.empty(4, dtype=float)

with open(DNN_mean_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)

    header1 = next(csv_reader)
    header2 = next(csv_reader)

    data1 = [float(x) for x in header1[1:]]
    data2 = [float(x) for x in header2[1:]]

    mean_input = np.array(data1)
    std_input = np.array(data2)
# %%
error_df = pd.DataFrame(columns=['Errore'])
idx_flat = pd.DataFrame(columns=['Idx_flat'])
idx_flat = []
num_rows = len(wave_trapz_old[:, 0])

aux1 = np.arange(0, 1023, 8)
aux2 = np.linspace(0, 1023, 1024)

for num in tqdm(range(num_rows),desc="Processing", unit="iteration",colour='green'):

    wave_trapz = np.interp(aux1,aux2,wave_trapz_old[num,:])
    temp = aux_trapz.loc[num, 'T']
    freq = aux_trapz.loc[num, 'f']
    B_pkpk = aux_trapz.loc[num, 'pk_pk'] * 1e-3
    T = 1 / freq
    P_real = aux_trapz.loc[num, 'P']

    dt = T / len(wave_trapz)
    der_B = np.diff(wave_trapz) / dt
    f_eq_orig = 0.5 * np.abs(der_B) / B_pkpk

    aux_l = np.where(f_eq_orig < 50e3)
    f_eq=f_eq_orig.copy()

    f_l = f_eq[aux_l].copy()
    f_eq[aux_l] = 50e3
    ciccio = list(filter(lambda x: x < 5e3, f_eq_orig))
    if len(ciccio)>5:
        idx_flat.append(num)
        continue

    input = pd.DataFrame()
    input['f'] = f_eq.T
    input['pk_pk'] = 1e3 * B_pkpk  # *np.ones(np.shape(f_eq.T))
    input['T'] = temp  # *np.ones(np.shape(f_eq.T))
    input['duty'] = 0.5

    col = ['f', 'pk_pk']
    input[col] = input[col].apply(lambda x: np.log10(x))

    input = (input - mean_input) / std_input

    tri_sim_pred = 1e-3 * 10 ** model.predict(input,verbose=0)

    trapz_pred = freq ** 2 * np.sum(tri_sim_pred) * dt

    l = []
    l.append(tri_sim_pred)

    error = 100 * (trapz_pred - P_real) / P_real

    error_df = pd.concat([error_df, pd.DataFrame({'Errore': [error]})], ignore_index=True)


print(l[:25])



# plt.figure()
# hist_select = error_df.loc[list(set(range(error_df.shape[0])).difference(idx_flat)), :]
# plt.hist(np.abs(hist_select),bins=20)
# plt.show()
    
# with open("Dataset_trapz_N87_mod.json", 'w') as file:
#     data = {}
#     data['Flux_Density'] = input['pk_pk'].to_numpy().tolist()
#     data['Frequency'] = input['f'].to_numpy().tolist()
#     data['Temperature'] = input['T'].to_numpy().tolist()
#     data['Duty_Ratio'] = input['duty'].to_numpy().tolist()
#     data['Power_Loss'] = trapz_pred.tolist()
#     json_data = json.dumps(data)
#     file.write(json_data)




