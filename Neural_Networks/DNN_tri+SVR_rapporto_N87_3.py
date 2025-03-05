import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from keras_tuner.src.backend.io import tf
import copy
import random
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from joblib import dump, load

from Aux_fun import db_read, reg_plot, rapporto_sin_tri

mat_type = 'N87_val'

aux_sin_pre, aux_tri_pre = db_read(mat_type, sym = False)

model = keras.models.load_model('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\DNN_tri_N87.keras')

file_name = "C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\Mean&Sd_tri_N87.csv"

svr = load('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\SVR_ratio_N87.joblib')

# Dichiara le variabili
mean_input = np.empty(4, dtype=float)
std_input = np.empty(4, dtype=float)

# Apri il file in modalità lettura
with open(file_name, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)

    # Leggi i nomi delle colonne
    header1 = next(csv_reader)
    header2 = next(csv_reader)

    # Leggi i valori delle variabili
    data1 = [float(x) for x in header1[1:]]
    data2 = [float(x) for x in header2[1:]]

    mean_input = np.array(data1)
    std_input = np.array(data2)


input = aux_sin_pre.loc[:,['f', 'pk_pk', 'T']].copy()
input['duty'] = 0.5
# Seleziona solo le colonne su cui desideri calcolare il logaritmo
col = ['f', 'pk_pk']
input[col] = input[col].apply(lambda x: np.log10(x))

input = (input - mean_input) / std_input

tri_sim_pred = model.predict(input)

aux_tri_new = aux_sin_pre.loc[:,['f', 'pk_pk', 'T']].copy()
aux_tri_new['duty'] = 0.5
aux_tri_new ['W']=10**tri_sim_pred
aux_tri_new['P'] = aux_tri_new['W'] * aux_tri_new['f']

aux_sin, aux_tri = rapporto_sin_tri(aux_sin_pre, aux_tri_new, plot=True)

# %%
aux = np.isfinite(aux_sin.loc[:, 'Rapporto_W'])
input = aux_sin.loc[aux, ['f', 'pk_pk', 'T']].copy()
# Seleziona solo le colonne su cui desideri calcolare il logaritmo
col = ['f', 'pk_pk']
# input[col] = input[col].apply(lambda x: np.log10(x))
# output=df_tri['W']/df_tri['f']
output = aux_sin.loc[aux, 'W']
output = output.to_frame(name='W')
# %%
# input, output = shuffle(input, output, random_state=1)


file_name = "C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\Mean&Sd_SVR_N87.csv"
# Dichiara le variabili
mean_input = np.empty(4, dtype=float)
std_input = np.empty(4, dtype=float)

# Apri il file in modalità lettura
with open(file_name, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)

    # Leggi i nomi delle colonne
    header1 = next(csv_reader)
    header2 = next(csv_reader)

    # Leggi i valori delle variabili
    data1 = [float(x) for x in header1[1:]]
    data2 = [float(x) for x in header2[1:]]

    mean_input = np.array(data1)
    std_input = np.array(data2)

X_test = (input - mean_input[0:3]) / std_input[0:3]

y_test = output

n_cols = X_test.shape[1]

yfit = svr.predict(X_test)

predictions = yfit.T*aux_sin.loc[:,'W_tri']

type = 'test'
reg_plot(y_test, predictions, type)

X_test_denorm = X_test * std_input[0:3] + mean_input[0:3]

P_pred=predictions*aux_sin.loc[:, 'f']/1e3

aux_sin['P_pred']=P_pred

Results = aux_sin[['f', 'pk_pk', 'T', 'P_pred', 'P']].copy()

Results.to_csv('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Results_2023_11_10\\N87_sin.csv', sep=',', index=False)

reg_plot(Results.loc[:,'P'], Results.loc[:,'P_pred'], type)

# Plotta gli output predetti rispetto agli input denormalizzati
plt.scatter(Results.loc[:,'pk_pk'], Results.loc[:,'P'], label='Valori Effettivi', color='blue', alpha=0.4)
plt.scatter(Results.loc[:,'pk_pk'], Results.loc[:,'P_pred'], label='Valori Predetti', color='red', alpha=0.6)
plt.yscale('log')
plt.xscale('log')
plt.grid()
# plt.axhline(y=np.pi**2/8, color='red', linestyle='-')
plt.xlabel('Input Denormalizzati')
plt.ylabel('Output')
plt.legend()
plt.title('Output Predetti vs. Input Denormalizzati (test Set)')
plt.show()

#
#
#
#
#
# # Plotta gli output predetti rispetto agli input denormalizzati
# plt.scatter(X_test_denorm.iloc[:, 1], aux_sin.loc[:,'P'], label='Valori Effettivi', color='blue', alpha=0.4)
# plt.scatter(X_test_denorm.iloc[:, 1], P_pred, label='Valori Predetti', color='red', alpha=0.6)
# plt.yscale('log')
# plt.xscale('log')
# plt.grid()
# # plt.axhline(y=np.pi**2/8, color='red', linestyle='-')
# plt.xlabel('Input Denormalizzati')
# plt.ylabel('Output')
# plt.legend()
# plt.title('Output Predetti vs. Input Denormalizzati (test Set)')
# plt.show()
# #
# # Converti y_pred in un DataFrame
# X_val_denorm.reset_index(drop=True, inplace=True)
# y_pred_df = pd.DataFrame(y_pred)
# df_svr = pd.DataFrame()
# df_svr = pd.concat([X_val_denorm, y_pred_df], axis=1,)
# df_svr.columns = ['f','B','T','Rapporto_W']
# sns.scatterplot(df_svr, x='f', y='B', hue='Rapporto_W')
# # plt.axhline(y=8/np.pi**2, color='red', linestyle='-')
# plt.xlabel('f')
# plt.ylabel('B')
# plt.title('B f')
# # plt.yscale('log')
# # plt.xscale('log')
# plt.grid(True)
# plt.show()
#
# input.reset_index(drop=True, inplace=True)
# output.reset_index(drop=True, inplace=True)
# df_base = pd.DataFrame()
# df_base = pd.concat([input, output], axis=1,)
# df_base.columns = ['f','B','T','Rapporto_W']
# sns.scatterplot(df_base, x='f', y='B', hue='Rapporto_W')
# # plt.axhline(y=8/np.pi**2, color='red', linestyle='-')
# plt.xlabel('f')
# plt.ylabel('B')
# plt.title('B f')
# # plt.yscale('log')
# # plt.xscale('log')
# plt.grid(True)
# plt.show()
#
# mse_svr = ((y_val.values.ravel() - y_pred) ** 2).mean()
# # Visualizza l'MSE
# print("Mean Squared Error (MSE) SVR (x1e3):", mse_svr*1e3)
#
# mse_8pi2 = ((y_val.values.ravel() - np.pi**2/8) ** 2).mean()
# # Visualizza l'MSE
# print("Mean Squared Error (MSE) 8/pi^2 (x1e3):", mse_8pi2*1e3)
#
# err_svr=((y_val.values.ravel() - y_pred) / y_val.values.ravel())*100
#
# err_8pi2=((y_val.values.ravel() - np.pi**2/8) / y_val.values.ravel())*100
#
# # Specifica la larghezza dei bin desiderata
# bin_width = 1
#
# # Calcola il numero di bin in base alla larghezza specificata
# bin_count = int((max(err_svr) - min(err_svr)) / bin_width)
# # Crea l'istogramma con la larghezza dei bin specificata
# plt.hist(err_svr, bins=bin_count, range=(min(err_svr), max(err_svr)), edgecolor='black')
# # plt.hist(err_svr, bins=20, edgecolor='black')  # Puoi regolare il numero di bin a tuo piacimento
# plt.axvline(x=0, color='red', linestyle='-')
# plt.xlabel('Errore %')
# plt.ylabel('Frequenza')
# plt.title('SVR')
# plt.show()
#
# bin_count = int((max(err_8pi2) - min(err_8pi2)) / bin_width)
# # Crea l'istogramma con la larghezza dei bin specificata
# plt.hist(err_8pi2, bins=bin_count, range=(min(err_8pi2), max(err_8pi2)), edgecolor='black')
# # plt.hist(err_8pi2, bins=20, edgecolor='black')  # Puoi regolare il numero di bin a tuo piacimento
# plt.axvline(x=0, color='red', linestyle='-')
# plt.xlabel('Errore %')
# plt.ylabel('Frequenza')
# plt.title('8/pi^2')
# plt.show()
#
# print("Errore max SVR: " + str(max(max(err_svr), abs(min(err_svr)))))
# print("Errore max 8/pi^2 " + str(max(max(err_8pi2), abs(min(err_8pi2)))))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# aux_sin, aux_tri = db_read(mat_type, sym = False)
# model = keras.models.load_model('Model/DNN_tri.keras')
# svr = load('Model\SVR.joblib')
# # Specifica il nome del file CSV
# file_name = "Model/Mean&Sd.csv"
# # Dichiara le variabili
# mean_input = np.empty(4, dtype=float)
# std_input = np.empty(4, dtype=float)
#
# # Apri il file in modalità lettura
# with open(file_name, 'r') as csvfile:
#     csv_reader = csv.reader(csvfile)
#
#     # Leggi i nomi delle colonne
#     header1 = next(csv_reader)
#     header2 = next(csv_reader)
#
#     # Leggi i valori delle variabili
#     data1 = [float(x) for x in header1[1:]]
#     data2 = [float(x) for x in header2[1:]]
#
#     mean_input = np.array(data1)
#     std_input = np.array(data2)
#
#
# input = aux_sin.loc[:,['f', 'pk_pk', 'T']].copy()
# input['duty'] = 0.5
# # Seleziona solo le colonne su cui desideri calcolare il logaritmo
# col = ['f', 'pk_pk']
# input[col] = input[col].apply(lambda x: np.log10(x))
# # output=df_sin['W']/df_sin['f']
# output = aux_sin.loc[:,'W']
# output=output.to_frame(name='W')
#
# input = (input - mean_input) / std_input
#
# tri_sim_pred = model.predict(input)
#
# rapporto_pred = svr.predict(input.loc[:,['f', 'pk_pk', 'T']].copy())
# W_pred = (rapporto_pred * (output.values).T).T
# W_pred = np.log10(W_pred)
# output['W'] = output['W'].apply(lambda x: np.log10(x))
# type = 'val'
# reg_plot(output, W_pred, type)
#
# input_denorm = input * std_input + mean_input
#
# # Plotta gli output predetti rispetto agli input denormalizzati
# plt.figure(figsize=(6, 6))
# plt.scatter(input_denorm.iloc[:, 1], output, label='Valori Effettivi', color='blue', alpha=0.4)
# plt.scatter(input_denorm.iloc[:, 1], W_pred, label='Valori Predetti', color='red', alpha=0.6)
# plt.xlabel('Input Denormalizzati')
# plt.ylabel('Output')
# plt.legend()
# plt.title('Output Predetti vs. Input Denormalizzati (Val Set)')
# plt.show()
