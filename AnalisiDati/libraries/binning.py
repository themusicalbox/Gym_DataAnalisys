'''
Il modulo contiene le funzioni necessarie al binning dei dati
'''
# importo le librerie necessarie
import numpy as np
import pandas as pd


def bin_data(dataframe, zones, n_bins):
    '''
    La funzione riceve tre argomenti:
        - Un DataFrame di Pandas formattato con due colonne [Vp, I]
        - Una lista degli estremi delle zone in cui binnare i dati
            ad esempio [np.min(data.Vp), -1, np.max(data.Vp)] per
            distinguere tra la zona lineare e la zona esponenziale
        - Una lista con il numero di bin per ogni zona ad esempio
            [100,1000]
    Viene ritornato un DataFrame di Pandas con 4 colonne:
    [Vp, I, errVp, errI]
    '''
    # per ovviare al problema degli elementi duplicati nei punti di
    # raccordo, devo fare in modo che il punto di stop venga incluso
    # solo per l'ultimo segmento di dati. Dunque introduco un
    # controllo con if.

    # TODO: inserire un check sui tipi inseriti

    # creo un array di numpy per contenere gli estremi dei bin
    bins = np.zeros(0)

    # riempio il vettore creando per ciascuna zona un inieme di punti
    # unifomrmemente distribuiti.
    # si noti che il ciclo usa la funizone range: range esclude l'estremo
    # superiore così che range(3) crei (0, 1, 2). Cosi facendo, ipotizando
    # di avere 3 zone, dunque con la lista zones lunga 4 il ciclo si fermerà
    # con i = 2, dunque l'ultimo posto letto dalla funzione sarà, i+1=3,
    # ovvero, il quarto elemento. La funzione esclude, per ogni zona meno che
    # l'ultima, l'estremo superiore, che verrà inserito come estremo inferiore
    # della zona successiva.
    for i in range(len(zones)-1):
        if (i+1) < (len(zones)-1):
            bins = np.append(bins, np.linspace(zones[i],
                                               zones[i+1],
                                               n_bins[i],
                                               endpoint=False
                                               )
                             )
        else:
            bins = np.append(bins, np.linspace(zones[i],
                                               zones[i+1],
                                               n_bins[i],
                                               endpoint=True
                                               )
                             )
    # ora che ho un vettore senza duplicati con tutti
    # gli intervalli per i bin posso raggruppare i dati
    group = dataframe.groupby(pd.cut(dataframe.Vp, bins))
    means = group.mean()[['Vp', 'I']]
    stds = group.std()[['Vp', 'I']]
    stds.columns = ['errVp', 'errI']
    binned = pd.concat([means, stds], axis=1)
    binned = binned.reset_index(drop=True)
    binned.dropna(inplace=True)
    return binned
