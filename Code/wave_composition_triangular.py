import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate

csv_B = 'N87\B_waveform[T].csv'
B_table = pd.read_csv(csv_B)

csv_f = 'N87/Frequency[Hz].csv'
f_list = pd.read_csv(csv_f)


csv_H = 'N87/H_waveform[Am-1].csv'
H_table = pd.read_csv(csv_H)

csv_W = 'N87/Volumetric_losses[Wm-3].csv'
W_list = pd.read_csv(csv_W)

# Parametri del segnale triangolare scaleno
amplitude_rise = 1.5  # Amplitude per la fase crescente
amplitude_fall = 1.5  # Amplitude per la fase decrescente
frequency = 100000       # Frequenza del segnale (numero di cicli al secondo)
duration = 0.00002       # Durata del segnale in secondi

# Creazione dell'array dei tempi
sample_rate = frequency*1024    # Frequenza di campionamento in Hz
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generazione del segnale triangolare scaleno
triangular_signal = np.zeros_like(t)
cycle_duration = 1.0 / frequency
for i, time in enumerate(t):
    phase = (time % cycle_duration) / cycle_duration
    if phase < 0.25:
        triangular_signal[i] = 2 * amplitude_rise * phase
        cc=triangular_signal[i]
    else:
        triangular_signal[i] =cc - (2 * amplitude_fall/3 * (phase - 0.25))

# Plot del segnale triangolare scaleno
plt.figure(figsize=(10, 6))
plt.plot(t, triangular_signal)
plt.title("Segnale Triangolare Scaleno")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza")
plt.grid(True)
plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#CWH
# Calcola la derivata utilizzando la differenza centrale
delta_x = t[1] - t[0]  # Passo di campionamento
derivative = np.gradient(triangular_signal, delta_x)
plt.plot(t, derivative)
plt.show()

print(type(derivative))

derivata_pos_count = 0
derivata_neg_count = 0

for i in range (1, len(derivative)):
    if derivative[i]> 0 :
        derivata_pos_count = derivata_pos_count +1
    elif derivative[i]< 0:
        derivata_neg_count = derivata_neg_count + 1

d = derivata_pos_count / (derivata_neg_count+ derivata_pos_count)
print(d)

f1 =frequency/(2*d)
f2 = frequency/(2*(1-d))
print(f1,f2)

f12=[f1,f2]

if f12[0]> 500000 or f12[1] > 500000:
    print("errore, frequenza troppo alta")
elif f12[0]< 50000 or f12[1] < 50000:
    print("errore, frequenza troppo bassa")
else:
    print("ci")






#if f1> 500000 or f2 > 500000:
    #print("errore, frequenza troppo alta")
#elif f1< 50000 or f2 < 50000:
    #print("errore, frequenza troppo bassa")
#else:
    #print("ci")

indici = []
frequenze_appros = []
potenze_appros = []

array_f = f_list.to_numpy()  #conversione in array
array_W = W_list.to_numpy()

for i in range(2):
    differenze = np.abs(array_f- f12[i])
    indice = np.argmin(differenze)
    f_appros = array_f[indice]
    frequenze_appros.append(f_appros)
    indici.append(indice)
    potenze= array_W[indice]
    potenze_appros.append(potenze)

Perdite_tot = frequency*(potenze_appros[0]/f12[0] + potenze_appros[1]/f12[1])


print(indici)
print(frequenze_appros)
print(potenze_appros, Perdite_tot)
