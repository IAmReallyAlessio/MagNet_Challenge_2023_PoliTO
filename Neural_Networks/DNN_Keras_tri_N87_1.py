import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
import keras_tuner
from keras_tuner.src.backend.io import tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Aux_fun import db_read, reg_plot

#%%

# Chiedi all'utente di inserire un valore da tastiera e assegnalo a una variabile
# train = input("Desideri addestrare la rete? (True/False): ")
train = 'True'
opt = 'True'
save = 'False'


if train == 'False':
    save = 'False'
    model = keras.models.load_model('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\DNN_tri_opt.keras')
    # Specifica il nome del file CSV
    file_name = "C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Mean&Sd_opt_11_08.csv"
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

elif train == 'True':
    # epochs = input("Definire il numero di iterazioni: ")
    # epochs = int(epochs)
    # save = input("Desideri salvare la rete addestrata (True/False): ")
    # mat_type = input("Definire il materiale (N87_train): ")
    mat_type = 'N87_train'
    epochs = 10000
    # mat_type = 'N87_train'
    aux_sin, aux_tri = db_read(mat_type, sym = False)



    #%%
    # aux = np.isfinite(aux_sin.loc[:,'Rapporto_W'])
    input = aux_tri.loc[:,['f', 'pk_pk', 'T', 'duty']].copy()
    # Seleziona solo le colonne su cui desideri calcolare il logaritmo
    col = ['f', 'pk_pk']
    input[col] = input[col].apply(lambda x: np.log10(x))
    # output=df_sin['W']/df_sin['f']
    output = aux_tri.loc[:,'W']
    output=output.to_frame(name='W')
    output['W'] = output['W'].apply(lambda x: np.log10(x))
    #%%
    input, output = shuffle(input, output, random_state=1)

    # # Prima, dividiamo in set di addestramento e test (90% addestramento, 10% test)
    # X_train, X_test, y_train, y_test = train_test_split(input_norm, output, test_size=0.01, random_state=1)
    X_train = input
    y_train = output

    # Ora dividiamo il set di addestramento in addestramento e validazione (80% addestramento, 20% validazione)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)#, random_state=1)

    mean_input = X_train.mean()
    std_input = X_train.std()

    X_train = (X_train - mean_input) / std_input
    X_val = (X_val - mean_input) / std_input

    # Ora hai X_train, y_train per addestramento, X_val, y_val per validazione e X_test, y_test per test

    n_cols = X_train.shape[1]

    # define regression model
    def regression_model():
        # create model
        model = Sequential()
        model.add(Dense(16, activation='tanh', input_shape=(n_cols,)))
        # model.add(Dense(36, activation='tanh'))
        model.add(Dense(12, activation='tanh'))
        model.add(Dense(4, activation='tanh'))
        # model.add(Dense(3, activation='tanh'))
        model.add(Dense(1))

        # compile model
        model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_squared_error'])
        return model

    def opt_model(hp):
        # create model
        model = Sequential()
        model.add(Dense(12, activation=hp.Choice("activation", ["relu", "tanh"]), input_shape=(n_cols,)))
        for i in range(hp.Int("num_layers", 0, 2)):
            model.add(
                Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=4, max_value=12, step=4),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
        model.add(Dense(1))
        # compile model
        model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_squared_error'])
        return model

    # build the model
    # model = regression_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=1, restore_best_weights=True)

    if opt == 'True':
        tuner = keras_tuner.RandomSearch(
            hypermodel=opt_model,
            objective="val_loss",
            max_trials=30,
            executions_per_trial=1,
            overwrite=False,
            directory="keras_tuner",
            project_name="prova_tuner_11_08",
        )

        tuner.search(X_train, y_train, epochs=200, validation_data=(X_val, y_val), verbose=1)
        tuner.results_summary()
        best_model = tuner.get_best_models(num_models=1)
        model = best_model[0]
        model.build()
        model.summary()
    # # fit the model
    else:
        # build the model
        # model = regression_model()
        model = keras.models.load_model('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model/DNN_tri_opt.keras')


    #%%
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=1,
                        callbacks=[callback])
    # Estrai i valori di perdita (loss) per il set di addestramento e il set di validazione dall'oggetto 'history'
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Crea un grafico dei valori di perdita
    plt.figure(figsize=(6, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs (Log Scale)')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    # plt.xscale('log')
    plt.legend()
    plt.title('Training and Validation Loss (Bilogarithmic Scale)')
    plt.grid()  # Aggiungi griglia per una migliore leggibilità

    # Addestramento
    train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
    print(f'Errore quadratico medio sul set di addestramento (x1e3): {1000*train_loss:.4f}')

    # Validazione
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
    print(f'Errore quadratico medio sul set di validazione (x1e3): {1000*val_loss:.4f}')

    predictions = model.predict(X_val)
    type = 'val'
    reg_plot(y_val, predictions, type)

    X_val_denorm = X_val * std_input + mean_input
    y_pred = model.predict(X_val)

    # Plotta gli output predetti rispetto agli input denormalizzati
    plt.figure(figsize=(6, 6))
    plt.scatter(X_val_denorm.iloc[:, 1], y_val, label='Valori Effettivi', color='blue', alpha=0.4)
    plt.scatter(X_val_denorm.iloc[:, 1], y_pred, label='Valori Predetti', color='red', alpha=0.6)
    plt.xlabel('Input Denormalizzati')
    plt.ylabel('Output')
    plt.legend()
    plt.title('Output Predetti vs. Input Denormalizzati (Val Set)')

mat_type = 'N87_val'
aux_sin, aux_tri = db_read(mat_type, sym = False)
# aux = np.isfinite(aux_sin.loc[:,'Rapporto_W'])
input = aux_tri.loc[:,['f', 'pk_pk', 'T', 'duty']].copy()
# Seleziona solo le colonne su cui desideri calcolare il logaritmo
col = ['f', 'pk_pk']
input[col] = input[col].apply(lambda x: np.log10(x))
# output=df_sin['W']/df_sin['f']
output = aux_tri.loc[:,'W']
output=output.to_frame(name='W')
output['W'] = output['W'].apply(lambda x: np.log10(x))

input = (input - mean_input) / std_input

val_set_loss = model.evaluate(input, output, verbose=0)[0]
print(f'Errore quadratico medio sul set di validazione (new) (x1e3):: {1000*val_set_loss:.4f}')

predictions = model.predict(input)
type = 'val new'
reg_plot(output, predictions, type)

input_denorm = input * std_input + mean_input
y_pred = model.predict(input)

# Plotta gli output predetti rispetto agli input denormalizzati
plt.figure(figsize=(6, 6))
plt.scatter(input_denorm.iloc[:, 1], output, label='Valori Effettivi', color='blue', alpha=0.4)
plt.scatter(input_denorm.iloc[:, 1], y_pred, label='Valori Predetti', color='red', alpha=0.6)
plt.xlabel('Input Denormalizzati')
plt.ylabel('Output')
plt.legend()
plt.title('Output Predetti vs. Input Denormalizzati (Val Set New)')

# plt.show()


# #%%
# if save == 'True':
#     model.save('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\\Ok\\DNN_tri_opt_red_N87.keras')
#
#     # Specifica il nome del file CSV
#     file_name = "C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Model\Ok\\Mean&Sd_tri_opt_red_N87.csv"
#
#     # Apri il file in modalità scrittura
#     with open(file_name, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#
#         # Scrivi le variabili nel file CSV in colonne
#         csv_writer.writerow(['Mean'] + mean_input.tolist())
#         csv_writer.writerow(['Sd'] + std_input.tolist())
# # #%%
# #
# # aux_sin_pre, aux_tri_pre = db_read(mat_type, sym = True)
# # aux_sin, aux_tri = rapporto_tri_sin(aux_sin_pre, aux_tri_pre, plot=False)
#
# #%%
#
# P_pred=10**y_pred*aux_tri.loc[:, 'f']/1e3
#
# aux_tri['P_pred']=P_pred
#
# Results = aux_tri[['f', 'pk_pk', 'T', 'duty', 'P_pred', 'P']].copy()
#
# Results.to_csv('C:\\Users\\Cadema\\PycharmProjects\\Mag_Net_pvt\\Results_2023_11_10\\N87_tri_mio_mod.csv', sep=',', index=False)
