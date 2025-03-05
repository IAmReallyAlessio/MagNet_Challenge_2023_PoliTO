import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import csv

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from Aux_fun import reg_plot, db_read, rapporto_tri_sin, rapporto_sin_tri

#%%
# mat_type = input("Definire il materiale (N87_train): ")
mat_type = 'N87_train'
aux_sin_pre, aux_tri_pre = db_read(mat_type, sym = True)

model = keras.models.load_model('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\DNN_tri_N87.keras')

file_name = "C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\Mean&Sd_tri_N87.csv"
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
# output=df_sin['W']/df_sin['f']
# output = aux_tri.loc[:,'W']
# output=output.to_frame(name='W')

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
output = aux_sin.loc[aux, 'Rapporto_W']
output = output.to_frame(name='Rapporto_W')
# %%
input, output = shuffle(input, output, random_state=1)

# # Prima, dividiamo in set di addestramento e test (90% addestramento, 10% test)
# X_train, X_test, y_train, y_test = train_test_split(input_norm, output, test_size=0.01, random_state=1)
X_train = input
y_train = output

# Ora dividiamo il set di addestramento in addestramento e validazione (80% addestramento, 20% validazione)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

mean_input = X_train.mean()
std_input = X_train.std()

X_train = (X_train - mean_input) / std_input
X_val = (X_val - mean_input) / std_input

# Ora hai X_train, y_train per addestramento, X_val, y_val per validazione e X_test, y_test per test

n_cols = X_train.shape[1]

param_opt = False


if param_opt == True:
    param_grid = {'C': np.linspace(300,500,6), 'gamma': np.linspace(0.1,0.2,6),'epsilon': np.linspace(0.01,0.02,6)}
    grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=2,n_jobs=-1, scoring='r2')
    grid.fit(X_train,y_train.values.ravel())
    print(grid.best_estimator_)

if param_opt == True:
    svr=grid
# define regression model
else:
    svr = SVR(kernel="rbf", C=500, epsilon=0.012, gamma=0.13, verbose=True) #C=185, epsilon=0.014, gamma=0.1
    svr.fit(X_train, y_train.values.ravel())

dump(svr, 'C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\SVR_ratio_N87.joblib')

yfit = svr.predict(X_train)

score = svr.score(X_train,y_train)
print("R-squared:", score)
print("MSE:", mean_squared_error(y_train, yfit))

predictions = svr.predict(X_val)
type = 'val'
reg_plot(y_val, predictions, type)

X_val_denorm = X_val * std_input[0:3] + mean_input[0:3]
y_pred = svr.predict(X_val)

# Plotta gli output predetti rispetto agli input denormalizzati
plt.scatter(X_val_denorm.iloc[:, 1], y_val, label='Valori Effettivi', color='blue', alpha=0.4)
plt.scatter(X_val_denorm.iloc[:, 1], y_pred, label='Valori Predetti', color='red', alpha=0.6)
plt.axhline(y=np.pi**2/8, color='red', linestyle='-')
plt.xlabel('Input Denormalizzati')
plt.ylabel('Output')
plt.legend()
plt.title('Output Predetti vs. Input Denormalizzati (Val Set)')
plt.show()

# Converti y_pred in un DataFrame
X_val_denorm.reset_index(drop=True, inplace=True)
y_pred_df = pd.DataFrame(y_pred)
df_svr = pd.DataFrame()
df_svr = pd.concat([X_val_denorm, y_pred_df], axis=1,)
df_svr.columns = ['f','B','T','Rapporto_W']
sns.scatterplot(df_svr, x='f', y='B', hue='Rapporto_W')
# plt.axhline(y=8/np.pi**2, color='red', linestyle='-')
plt.xlabel('f')
plt.ylabel('B')
plt.title('B f')
# plt.yscale('log')
# plt.xscale('log')
plt.grid(True)
plt.show()

input.reset_index(drop=True, inplace=True)
output.reset_index(drop=True, inplace=True)
df_base = pd.DataFrame()
df_base = pd.concat([input, output], axis=1,)
df_base.columns = ['f','B','T','Rapporto_W']
sns.scatterplot(df_base, x='f', y='B', hue='Rapporto_W')
# plt.axhline(y=8/np.pi**2, color='red', linestyle='-')
plt.xlabel('f')
plt.ylabel('B')
plt.title('B f')
# plt.yscale('log')
# plt.xscale('log')
plt.grid(True)
plt.show()

mse_svr = ((y_val.values.ravel() - y_pred) ** 2).mean()
# Visualizza l'MSE
print("Mean Squared Error (MSE) SVR (x1e3):", mse_svr*1e3)

mse_8pi2 = ((y_val.values.ravel() - np.pi**2/8) ** 2).mean()
# Visualizza l'MSE
print("Mean Squared Error (MSE) 8/pi^2 (x1e3):", mse_8pi2*1e3)

err_svr=((y_val.values.ravel() - y_pred) / y_val.values.ravel())*100

err_8pi2=((y_val.values.ravel() - np.pi**2/8) / y_val.values.ravel())*100

# Specifica la larghezza dei bin desiderata
bin_width = 1

# Calcola il numero di bin in base alla larghezza specificata
bin_count = int((max(err_svr) - min(err_svr)) / bin_width)
# Crea l'istogramma con la larghezza dei bin specificata
plt.hist(err_svr, bins=bin_count, range=(min(err_svr), max(err_svr)), edgecolor='black')
# plt.hist(err_svr, bins=20, edgecolor='black')  # Puoi regolare il numero di bin a tuo piacimento
plt.axvline(x=0, color='red', linestyle='-')
plt.xlabel('Errore %')
plt.ylabel('Frequenza')
plt.title('SVR')
plt.show()

bin_count = int((max(err_8pi2) - min(err_8pi2)) / bin_width)
# Crea l'istogramma con la larghezza dei bin specificata
plt.hist(err_8pi2, bins=bin_count, range=(min(err_8pi2), max(err_8pi2)), edgecolor='black')
# plt.hist(err_8pi2, bins=20, edgecolor='black')  # Puoi regolare il numero di bin a tuo piacimento
plt.axvline(x=0, color='red', linestyle='-')
plt.xlabel('Errore %')
plt.ylabel('Frequenza')
plt.title('8/pi^2')
plt.show()

print("Errore max SVR: " + str(max(max(err_svr), abs(min(err_svr)))))
print("Errore max 8/pi^2 " + str(max(max(err_8pi2), abs(min(err_8pi2)))))

save = 'True'

if save == 'True':
    # Specifica il nome del file CSV
    file_name = "C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\Ok\\Mean&Sd_SVR_N87.csv"

    # Apri il file in modalità scrittura
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Scrivi le variabili nel file CSV in colonne
        csv_writer.writerow(['Mean'] + mean_input.tolist())
        csv_writer.writerow(['Sd'] + std_input.tolist())
# #%%
#
# aux_sin_pre, aux_tri_pre = db_read(mat_type, sym = True)
# aux_sin, aux_tri = rapporto_tri_sin(aux_sin_pre, aux_tri_pre, plot=False)