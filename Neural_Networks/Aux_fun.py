import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os

def db_read(mat_type, sym):
    current_file_path = os.path.abspath(__file__)
    basepath = os.path.dirname(os.path.dirname(current_file_path))
    data_path = os.path.join(basepath, 'data\\')
    df = pd.DataFrame()
    df1 = pd.read_csv(data_path + mat_type + '/B_waveform[T].csv')
    df['f'] = pd.read_csv(data_path + mat_type + '/Frequency[Hz].csv')
    df['T'] = pd.read_csv(data_path + mat_type + '/Temperature[C].csv')
    df['P'] = pd.read_csv(data_path + mat_type + '/Volumetric_losses[Wm-3].csv')
    df['W'] = df['P'] / df['f']


    # Calcola il valore picco-piccio su ciascuna riga
    peak_to_peak = np.ptp(df1, axis=1)
    # Calcola il valore efficace su ciascuna riga
    rms = np.sqrt(np.mean(df1 ** 2, axis=1))
    arv = np.mean(abs(df1), axis=1)
    fdc = peak_to_peak / 2 / rms
    fdf = rms / arv

    df['pk_pk'] = 1e3 * peak_to_peak
    df['rms'] = 1e3 * rms
    df['fdc'] = fdc
    df['fdf'] = fdf

    df['wave'] = 'other'
    # Condizione per selezionare le righe
    condizione_tri = (np.isclose(df['fdf'], 2 / (3 ** 0.5), rtol=0.05)) & (np.isclose(df['fdc'], 3 ** 0.5, rtol=0.05))
    condizione_sin = (np.isclose(df['fdf'], np.pi / (2 * 2 ** 0.5), rtol=0.006)) & (
        np.isclose(df['fdc'], 2 ** 0.5, rtol=0.006))
    # Assegna 'tri' alle righe che soddisfano la condizione
    df.loc[condizione_tri, 'wave'] = 'tri'
    df.loc[condizione_sin, 'wave'] = 'sin'
    # Ottieni gli indici delle righe che soddisfano la condizione
    indici_righe_tri = df.index[condizione_tri].tolist()
    # Ottieni gli indici delle righe che soddisfano la condizione
    indici_righe_sin = df.index[condizione_sin].tolist()

    df['W'] = 1e3 * df['W']

    df_sin = df[df['wave'] == 'sin'].copy()
    # df_sin[['f', 'P', 'W', 'pk_pk']] = np.log10(df_sin[['f', 'P', 'W', 'pk_pk']])

    df_tri = df[df['wave'] == 'tri'].copy()
    # df_tri[['f', 'P', 'W', 'pk_pk']] = np.log10(df_tri[['f', 'P', 'W', 'pk_pk']])

    dB_wf_tri = np.diff(df1.loc[indici_righe_tri, :])
    duty = (np.count_nonzero(dB_wf_tri > 0, axis=1) / len(dB_wf_tri[0]))
    df_tri['duty'] = duty

    # aux_sin = df_sin.loc[df_sin['T']==25]
    # aux_tri = df_tri.loc[(df_tri['T']==25) & (np.isclose(df_tri['duty'],0.5,atol = 0.003))]

    aux_sin = df_sin.copy()
    if sym == True:
        aux_tri = df_tri.loc[(np.isclose(df_tri['duty'], 0.5, atol=0.003))]
    else:
        aux_tri = df_tri.copy()

    aux_sin.reset_index(drop=True, inplace=True)
    aux_tri.reset_index(drop=True, inplace=True)
    return aux_sin, aux_tri

def db_read_trapz(mat_type):
    current_file_path = os.path.abspath(__file__)
    basepath = os.path.dirname(current_file_path)
    data_path = os.path.join(basepath, 'data\\')
    df = pd.DataFrame()
    df1 = pd.read_csv(data_path + mat_type + '\\B_waveform[T].csv')
    df['f'] = pd.read_csv(data_path + mat_type + '\\Frequency[Hz].csv')
    df['T'] = pd.read_csv(data_path + mat_type + '\\Temperature[C].csv')
    df['P'] = pd.read_csv(data_path + mat_type + '\\Volumetric_losses[Wm-3].csv')
    df['W'] = df['P'] / df['f']


    # Calcola il valore picco-picco su ciascuna riga
    peak_to_peak = np.ptp(df1, axis=1)
    # Calcola il valore efficace su ciascuna riga
    rms = np.sqrt(np.mean(df1 ** 2, axis=1))
    arv = np.mean(abs(df1), axis=1)
    fdc = peak_to_peak / 2 / rms
    fdf = rms / arv

    df['pk_pk'] = 1e3 * peak_to_peak
    df['rms'] = 1e3 * rms
    df['fdc'] = fdc
    df['fdf'] = fdf

    df['wave'] = 'trapz'
    # Condizione per selezionare le righe
    condizione_tri = (np.isclose(df['fdf'], 2 / (3 ** 0.5), rtol=0.05)) & (np.isclose(df['fdc'], 3 ** 0.5, rtol=0.05))
    condizione_sin = (np.isclose(df['fdf'], np.pi / (2 * 2 ** 0.5), rtol=0.006)) & (
        np.isclose(df['fdc'], 2 ** 0.5, rtol=0.006))
    # Assegna 'tri' alle righe che soddisfano la condizione
    df.loc[condizione_tri, 'wave'] = 'tri'
    df.loc[condizione_sin, 'wave'] = 'sin'

    indici = set(df.index)
    # Ottieni gli indici delle righe che soddisfano la condizione
    indici_righe_tri = df.index[condizione_tri].tolist()
    # Ottieni gli indici delle righe che soddisfano la condizione
    indici_righe_sin = df.index[condizione_sin].tolist()
    indici_righe_trapz = indici - set(indici_righe_tri) - set(indici_righe_sin)
    indici_righe_trapz = list(indici_righe_trapz)
    df['W'] = 1e3 * df['W']

    df_sin = df[df['wave'] == 'sin'].copy()
    # df_sin[['f', 'P', 'W', 'pk_pk']] = np.log10(df_sin[['f', 'P', 'W', 'pk_pk']])

    df_tri = df[df['wave'] == 'tri'].copy()
    # df_tri[['f', 'P', 'W', 'pk_pk']] = np.log10(df_tri[['f', 'P', 'W', 'pk_pk']])

    df_trapz = df[df['wave'] == 'trapz'].copy()

    dB_wf_tri = np.diff(df1.loc[indici_righe_tri, :])
    duty = (np.count_nonzero(dB_wf_tri > 0, axis=1) / len(dB_wf_tri[0]))
    df_tri['duty'] = duty

    # aux_sin = df_sin.loc[df_sin['T']==25]
    # aux_tri = df_tri.loc[(df_tri['T']==25) & (np.isclose(df_tri['duty'],0.5,atol = 0.003))]

    aux_sin = df_sin.copy()
    aux_tri = df_tri.copy()
    aux_trapz = df_trapz.copy()
    aux_sin.reset_index(drop=True, inplace=True)
    aux_tri.reset_index(drop=True, inplace=True)
    aux_trapz.reset_index(drop=True, inplace=True)

    df_wf_trapz = (df1.loc[indici_righe_trapz, :]).to_numpy()

    # # Calcola il valore picco-picco su ciascuna riga
    # peak_to_peak = np.ptp(df_wf_trapz, axis=1)
    # # Calcola il valore efficace su ciascuna riga
    # rms = np.sqrt(np.mean(df_wf_trapz ** 2, axis=1))
    # arv = np.mean(abs(df_wf_trapz), axis=1)
    # fdc = peak_to_peak / 2 / rms
    # fdf = rms / arv
    #
    # df_check=pd.DataFrame()
    # df_check['pk_pk'] = 1e3 * peak_to_peak
    # df_check['rms'] = 1e3 * rms
    # df_check['fdc'] = fdc
    # df_check['fdf'] = fdf

    return aux_trapz, df_wf_trapz


def reg_plot(y_true, predictions, type):
    # Calcola il coefficiente R²
    r2 = r2_score(y_true, predictions)

    # Crea un modello di regressione lineare per plottare la retta
    linear_reg = LinearRegression()
    linear_reg.fit(y_true.values.reshape(-1, 1), predictions)

    # Ottieni i coefficienti della retta di regressione
    slope = linear_reg.coef_[0]
    intercept = linear_reg.intercept_

    # Plotta i risultati delle previsioni rispetto ai valori effettivi
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, predictions, color='blue', alpha=0.5)
    plt.xlabel('Valori Effettivi')
    plt.ylabel('Previsioni')
    plt.title(f'Regressione sul Set di {type} (R² = {r2:.2f})')
    # Plotta la retta di regressione
    plt.plot(y_true, slope * y_true + intercept, color='red', linewidth=2)
    plt.show()

def rapporto_tri_sin(aux_sin, aux_tri, plot):
    for index, row in aux_tri.iterrows():
        f_tri = row['f']
        B_tri = row['pk_pk']
        T_tri = row['T']

        # Filtra gli elementi di aux_tri che rientrano nella tolleranza specificata
        elementi_simili_f = aux_sin[
            (aux_sin['f'] >= (f_tri-10e3)) & (aux_sin['f'] <= (f_tri+10e3)) & (aux_sin['T'] == T_tri)].copy()
        # Calcola le differenze tra B_sin e tutti gli elementi nella colonna desiderata (ad esempio, 'B_column')
        if not elementi_simili_f.empty:
            elementi_simili_f.loc[:, 'differenza'] = np.abs(elementi_simili_f['pk_pk'].values - B_tri)
            # Trova l'indice del valore con la minima differenza
            if (elementi_simili_f['differenza'] <= 2).any():
                indice_minima_differenza = elementi_simili_f['differenza'].idxmin()
                aux_tri.at[index, 'f_sin'] = aux_sin.at[indice_minima_differenza, 'f']
                aux_tri.at[index, 'B_sin'] = aux_sin.at[indice_minima_differenza, 'pk_pk']
                aux_tri.at[index, 'T_sin'] = aux_sin.at[indice_minima_differenza, 'T']
                aux_tri.at[index, 'P_sin'] = aux_sin.at[indice_minima_differenza, 'P']
                aux_tri.at[index, 'W_sin'] = aux_sin.at[indice_minima_differenza, 'W']

    aux_tri.loc[:, 'W_corr'] = aux_tri.loc[:, 'W_sin'] * (aux_tri.loc[:, 'f']) / (
        aux_tri.loc[:, 'f_sin']) *(aux_tri.loc[:,'pk_pk']**2)/(aux_tri.loc[:,'B_sin']**2)
    aux_tri.loc[:, 'Rapporto_W'] = (aux_tri.loc[:, 'W_corr'] / aux_tri.loc[:, 'W'])

    # Estrai la colonna 'Rapport_W' dal DataFrame
    rapporto_w = aux_tri['Rapporto_W']
    # Crea l'istogramma
    plt.figure(figsize=(6, 6))
    plt.hist(rapporto_w, bins=20, edgecolor='black')  # Puoi regolare il numero di bin a tuo piacimento
    plt.axvline(x=np.pi ** 2 / 8, color='red', linestyle='-')
    plt.xlabel('Rapporto_W')
    plt.ylabel('Frequenza')
    plt.title('Istogramma di Rapport_W')
    plt.show()

    # np.count_nonzero(~np.isnan(rapporto_w))
    # rnd = random.sample(indici_righe_tri, 1000)
    # plt.plot(df1.iloc[rnd,:].T/np.max(df1.iloc[rnd,:], axis=1))
    # plt.show()

    if plot == True:
        plt.figure(figsize=(6, 6))
        plt.scatter(aux_tri['pk_pk'], aux_tri['Rapporto_W'])
        plt.axhline(y=np.pi ** 2 / 8, color='red', linestyle='-')
        plt.xlabel('B')
        plt.ylabel('W')
        plt.title('B W')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.grid(True)
        plt.show()
        # %%
        plt.figure(figsize=(6, 6))
        sns.scatterplot(aux_tri, x='f', y='pk_pk', hue='Rapporto_W')
        # plt.axhline(y=8/np.pi**2, color='red', linestyle='-')
        plt.xlabel('f')
        plt.ylabel('B')
        plt.title('B f')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.grid(True)
        plt.show()

        # #%%
        # df_plot=aux_sin.loc[np.isclose(df_tri['pk_pk'],150, rtol=0.1)]
        # sns.scatterplot(df_plot, x='f', y='Rapporto_W', hue='T', palette='tab10')
        # plt.xlabel('log f')
        # plt.ylabel('log W')
        # plt.title('f W')
        # # plt.yscale('log')
        # # plt.xscale('log')
        # plt.grid(True)
        # plt.show()
    return aux_sin, aux_tri


def rapporto_sin_tri(aux_sin, aux_tri, plot):
    for index, row in aux_sin.iterrows():
        f_sin = row['f']
        B_sin = row['pk_pk']
        T_sin = row['T']

        # Filtra gli elementi di aux_tri che rientrano nella tolleranza specificata
        elementi_simili_f = aux_tri[
            (aux_tri['f'] >= (f_sin-1e3)) & (aux_tri['f'] <= (f_sin +1e3)) & (aux_tri['T'] == T_sin)].copy()
        # Calcola le differenze tra B_sin e tutti gli elementi nella colonna desiderata (ad esempio, 'B_column')
        if not elementi_simili_f.empty:
            elementi_simili_f.loc[:, 'differenza'] = np.abs(elementi_simili_f['pk_pk'].values - B_sin)
            # Trova l'indice del valore con la minima differenza
            if (elementi_simili_f['differenza'] <= 1).any():
                indice_minima_differenza = elementi_simili_f['differenza'].idxmin()
                aux_sin.at[index, 'f_tri'] = aux_tri.at[indice_minima_differenza, 'f']
                aux_sin.at[index, 'B_tri'] = aux_tri.at[indice_minima_differenza, 'pk_pk']
                aux_sin.at[index, 'T_tri'] = aux_tri.at[indice_minima_differenza, 'T']
                aux_sin.at[index, 'P_tri'] = aux_tri.at[indice_minima_differenza, 'P']
                aux_sin.at[index, 'W_tri'] = aux_tri.at[indice_minima_differenza, 'W']

    aux_sin.loc[:, 'W_corr'] = (aux_sin.loc[:, 'W_tri'] * (aux_sin.loc[:, 'f']) /
                                (aux_sin.loc[:, 'f_tri']) *(aux_sin.loc[:,'pk_pk']**2)/(aux_sin.loc[:,'B_tri']**2))

    aux_sin.loc[:, 'Rapporto_W'] = (aux_sin.loc[:, 'W'] / aux_sin.loc[:, 'W_corr'])

    # Estrai la colonna 'Rapport_W' dal DataFrame
    rapporto_w = aux_sin['Rapporto_W']
    # Crea l'istogramma
    plt.figure(figsize=(6, 6))
    plt.hist(rapporto_w, bins=20, edgecolor='black')  # Puoi regolare il numero di bin a tuo piacimento
    plt.axvline(x=np.pi ** 2 / 8, color='red', linestyle='-')
    plt.xlabel('Rapporto_W')
    plt.ylabel('Frequenza')
    plt.title('Istogramma di Rapport_W')
    plt.show()

    # np.count_nonzero(~np.isnan(rapporto_w))
    # rnd = random.sample(indici_righe_tri, 1000)
    # plt.plot(df1.iloc[rnd,:].T/np.max(df1.iloc[rnd,:], axis=1))
    # plt.show()

    if plot == True:
        plt.figure(figsize=(6, 6))
        plt.scatter(aux_sin['pk_pk'], aux_sin['Rapporto_W'])
        plt.axhline(y=np.pi ** 2 / 8, color='red', linestyle='-')
        plt.xlabel('B')
        plt.ylabel('W')
        plt.title('B W')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.grid(True)
        plt.show()
        # %%
        plt.figure(figsize=(6, 6))
        sns.scatterplot(aux_sin, x='f', y='pk_pk', hue='Rapporto_W')
        # plt.axhline(y=8/np.pi**2, color='red', linestyle='-')
        plt.xlabel('f')
        plt.ylabel('B')
        plt.title('B f')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.grid(True)
        plt.show()

        # #%%
        # df_plot=aux_sin.loc[np.isclose(df_tri['pk_pk'],150, rtol=0.1)]
        # sns.scatterplot(df_plot, x='f', y='Rapporto_W', hue='T', palette='tab10')
        # plt.xlabel('log f')
        # plt.ylabel('log W')
        # plt.title('f W')
        # # plt.yscale('log')
        # # plt.xscale('log')
        # plt.grid(True)
        # plt.show()
    return aux_sin, aux_tri

if __name__ == "__main__":
    ciccio,pippo = db_read_trapz('N87_val')
