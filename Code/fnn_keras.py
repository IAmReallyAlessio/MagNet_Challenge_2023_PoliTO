import numpy as np
SEED = 5
np.random.seed(SEED)
import tensorflow
tensorflow.random.set_seed(SEED)

import keras
keras.utils.set_random_seed(SEED)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import keras_tuner
from keras_tuner.src.backend.io import tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import os
from pathlib import Path


NCOLS = 4
DATASET = pd.DataFrame()
EPOCHS = 5000

def fixed_model():
    model = Sequential()
    model.add(Dense(25, activation="relu", input_shape= (NCOLS,)))
    model.add(Dense(15, activation="tanh"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(2, activation="tanh"))
    #model.add(Dense(1))

    model.compile(optimizer = Adam(learning_rate=0.001447562812271149), loss = "mean_squared_error", metrics = ["mean_squared_error"])
    return model


def temp_model(hp):
    model = keras.Sequential()
    # Find the best number of layers and the best combination
    for i in range(2):
        model.add(
            Dense(
                units=hp.Int(f"units_{i}", min_value=2, max_value=25, step=1),
                activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
            )
        )
    model.add(Dense(units = 1, activation = hp.Choice("activation_fin", ["relu", "tanh"])))
    #learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["mean_squared_error"])
    return model
    
def dataset_creation():
    
    basepath = Path(__file__).parent
    # print(basepath)
    foldername = 'Training/Material_A'
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

    dB_pkpk = aux_dB.ptp(axis=1)
    dB_rms = np.sqrt(np.mean(aux_dB**2, axis=1))
    dB_arv = np.mean(abs(aux_dB), axis=1)
    k_f_dB = dB_rms/dB_arv

    aux_df = np.stack((B_pkpk, B_rms, B_arv, k_f_B, dB_pkpk, dB_rms, dB_arv, k_f_dB), axis=1)
    df = pd.DataFrame(aux_df, columns=['B_pkpk', 'B_rms', 'B_arv', 'k_f_B', 'dB_pkpk', 'dB_rms', 'dB_arv', 'k_f_dB'])
    df['Frequency'] = Frequency.values
    df['Power_Loss'] = Power_losses.values
    df['Temperature'] = Temperature.values

    df_tri = df.loc[(df['k_f_B'] > 2/(3**0.5)*0.9980) & (df['k_f_B'] < 2/(3**0.5)*1.002)]

    print(len(df_tri))

    tri_values = []
    for x in aux:
        B_rms = np.sqrt(np.mean(x ** 2, axis=0))
        B_arv = np.mean(abs(x), axis=0)
        k_f_B = B_rms / B_arv
        if k_f_B > 2/(3**0.5)*0.9980 and k_f_B < 2/(3**0.5)*1.002:
            tri_values.append(x)
    tri_values = np.array(tri_values)

    rms = np.sqrt(np.mean(tri_values**2, axis=1))
    duty_ratios = []
    for y in tri_values:
        # plt.plot(x, y_delta, '-')
        # plt.show()
        n_pos = 0
        n_neg = 0
        for j in range(1,1024):
            if y[j-1]>y[j]:
                n_pos += 1
            else:
                n_neg += 1
        duty_ratio = n_pos/(n_pos+n_neg)*10 #porto a intero
        duty_ratios.append(round(duty_ratio))
    duty_ratios = np.array(duty_ratios)

    rms = rms.tolist()
    t = []
    db = pd.DataFrame(columns = ["Flux", "Freq", "Temp", "Duty", "Power"])
    for tmp in rms:
        t.append(tmp*1e3)
    db['Flux'] = np.log10(t)
    db['Freq'] = np.log10(df_tri['Frequency'].to_numpy()).tolist()
    db['Temp'] = df_tri['Temperature'].to_numpy().tolist()
    db['Duty'] = duty_ratios.tolist()
    db['Power'] = np.log10(df_tri['Power_Loss'].to_numpy()).tolist()

    return db
    
def build_model():
    l1 = 13
    l2 = 8
    l3 = 17
    func = "tanh"

    return fixed_model(l1,l2,l3, func)

def execute():
    

    model = fixed_model()

    db = dataset_creation()

    input = db.loc[:,['Flux', 'Freq', 'Temp', 'Duty']].copy()
    output = db.loc[:, ["Power"]].copy()

    input, output = shuffle(input, output, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.1, random_state=1) #hold 10% of dataset for testing
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) #split the training dataset in 80% train 20% valid

    #implement EarlyStopping
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=1, restore_best_weights=True)
    
    #fit model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose=1, callbacks=[callback])

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    test_loss = model.evaluate(X_test, y_test)
    print(test_loss)

    


    plt.figure(figsize=(6, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs (Log Scale)')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    # plt.xscale('log')
    plt.legend()
    plt.title('Training and Validation Loss (Bilogarithmic Scale)')
    plt.grid()
    plt.show()


def tuner():
    db = dataset_creation()

    input_ = db.loc[:,['Flux', 'Freq', 'Temp', 'Duty']].copy()
    output_ = db.loc[:, ["Power"]].copy()

    input_, output_ = shuffle(input_, output_, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(input_, output_, test_size=0.1, random_state=1) #hold 10% of dataset for testing
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) #split the training dataset in 80% train 20% valid

    tmp = int(input("Si vuole fare una nuova run?(1=si, 0=no):"))
    if(tmp>0):
        tmp = input("Inserire il numero della run: ")
        tuner = keras_tuner.BayesianOptimization(
                                                    hypermodel=temp_model, 
                                                    objective=keras_tuner.Objective("val_loss", "min"), 
                                                    max_trials= 100, 
                                                    num_initial_points= 50, 
                                                    overwrite = False, 
                                                    directory = f"Run_{tmp}", 
                                                    project_name = "MagNet", 
                                                    executions_per_trial=5
                                                )
    else:
        tuner = keras_tuner.BayesianOptimization(hypermodel=temp_model, objective=keras_tuner.Objective("val_loss", "min"), max_trials= 100, num_initial_points= 50, overwrite = True)
    
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, verbose=1, restore_best_weights=True)
    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), verbose=2, callbacks = [callback])
    results = tuner.get_best_hyperparameters(3)

    fig, ax = plt.subplots(3,2)
    fig.set_size_inches(10,30)
    for i in range(0,3):
        model = temp_model(results[i])
        print(f"Training  {i+1}^ model:\n{results[i].values}")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, verbose=0, callbacks=[callback])

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        #model.save(f"./Run_{tmp}/model_{i+1}.keras")

        y_pred = model.predict(X_test)
        y_meas = y_test
        #print(y_pred)
        yy_pred = 10**(y_pred)
        yy_meas = 10**(y_meas.to_numpy())

        [test_loss, test_acc] = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {test_loss}\t|\t Test accuracy: {test_acc}")

        #print(yy_pred, yy_meas)
        # Relative Error
        Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
        Error_re_avg = np.mean(Error_re)
        Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
        Error_re_max = np.max(Error_re)
        Error_re_95th = np.percentile(Error_re, 95)
        print(f"Relative Error: {Error_re_avg:.8f}")
        print(f"RMS Error: {Error_re_rms:.8f}")
        print(f"MAX Error: {Error_re_max:.8f}")
        print(f"95th Percentile Error: {Error_re_95th:.8f}")

        ax[i][1].hist(Error_re)
        
        ax[i][0].semilogy(train_loss, label='Training Loss', color='blue')
        ax[i][0].semilogy(val_loss, label='Validation Loss', color='red')
        #ax1.yscale('log')
        # ax1.xscale('log')

        ax[i][0].grid()
    plt.show()   

def load():

    ans = input("From which run you need to load the model? (number only)")
    while(not ans.isnumeric()):
        ans = input("Number only! Insert Run number: ")
    
    run = int(ans)

    ans = input("Which model would you like to load? (1 to 3 where 1 is the best teoretically): ")
    while(not ans.isnumeric() and int(ans)>0 and int(ans)<4):
        ans = input("Number only! Insert model number: ")

    model = int(ans)

    model = keras.models.load_model(f"./Run_{run}/model_{model}.keras")
    model.summary()

    db = dataset_creation()

    input_ = db.loc[:,['Flux', 'Freq', 'Temp', 'Duty']].copy()
    output_ = db.loc[:, ["Power"]].copy()
    
    input_, output_ = shuffle(input_, output_, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(input_, output_, test_size=0.1, random_state=1) #use only 10% of dataset for testing

    y_pred = model.predict(X_test)
    y_meas = y_test
    #print(y_pred)
    yy_pred = 10**(y_pred)
    yy_meas = 10**(y_meas.to_numpy())

    #print(yy_pred, yy_meas)
    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    Error_re_avg = np.mean(Error_re)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
    Error_re_95th = np.percentile(Error_re, 95)
    print(f"Relative Error: {Error_re_avg:.8f}")
    print(f"RMS Error: {Error_re_rms:.8f}")
    print(f"MAX Error: {Error_re_max:.8f}")
    print(f"95th Percentile Error: {Error_re_95th:.8f}")
    plt.hist(Error_re)
    plt.show()

def main():
    
    while(True):
        ans = input("Tune, Load or Exec? ")
        if ans.lower() == "tune":
            tuner()
            break
        elif ans.lower() == "load":
            load()
            break
        elif ans.lower() == "exec":
            execute()
            break

if __name__ == "__main__":
    main()