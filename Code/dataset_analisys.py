import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from numpy.fft import fft, ifft

#%%
basepath = Path(__file__).parent   # prende il path corrente
foldername = 'N87'
filenames = os.listdir(basepath/foldername)

#---------------creazione-dataframes--------------------------------
db = pd.DataFrame()
filepath = basepath / foldername / filenames[0]
B_waveform = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0) #elenco valori di B
filepath = basepath / foldername / filenames[1]
Frequency = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0) #elenco frequenze
filepath = basepath / foldername / filenames[2]
H_waveform = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0) #elenco valori di H
filepath = basepath / foldername / filenames[3]
Temperature = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0) #elenco valori della Temperatura
filepath = basepath / foldername / filenames[4]
Power_losses = pd.concat((db, pd.read_csv(filepath, sep=',', header=None, )), axis=0) #elenco valori power loss

#%%
#---------setup----------------
""" rnd_state = 1
n = len(Frequency) """

#-------------------------------sampling--------------------------
""" B_waveform_red = B_waveform.sample(n=n, random_state=rnd_state)
Frequency_red = Frequency.sample(n=n, random_state=rnd_state)
Temperature_red = Temperature.sample(n=n, random_state=rnd_state)
Power_losses_red = Power_losses.sample(n=n, random_state=rnd_state) """
aux = B_waveform.to_numpy() 
Timestep = 1/(len(aux[0, :])*Frequency.to_numpy())
aux_dB = np.diff(aux)/Timestep

#----------------------------calcolo-valori-----------------------
B_pkpk = aux.ptp(axis=1)
B_rms = np.sqrt(np.mean(aux**2, axis=1)) #valore efficace
B_arv = np.mean(abs(aux), axis=1)        #valore medio
k_f_B = B_rms/B_arv                      #fattore di forma

#-----------------------creazione-dataframe--------------------------
aux_df = np.stack((B_pkpk, B_rms, B_arv, k_f_B), axis=1)
df = pd.DataFrame(aux_df, columns=['B_pkpk', 'B_rms', 'B_arv', 'k_f_B'])
df['Frequency'] = Frequency.values
df['Power_loss'] = Power_losses.values
df['Temperature'] = Temperature.values
""" output_file = open("df_print.txt", "w")
output_file.write(df.to_string()) """

#%%
#------------plot-grafico-onda------------------------
plt.plot(B_waveform.iloc[12])
plt.show
# %%
