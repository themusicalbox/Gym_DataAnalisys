# Questo file racchiude le funzioni utili a ripulire i dati
# che vengono tutte racchiuse nella funzione per leggere i 
# dati

# DEPENDENCIES: Python 3.7, Numpy, Pandas.

import numpy as np
import pandas as pd
from libraries import dtlg


def remove_starting(df, start_time):
    '''
    La funzione riceve un DataFrame ed elimina tutte le
    righe che corrispondono ad un indice di tempo (prima
    colonna) minore del secondo parametro passato alla funzione.
    '''
    # taglio tutto ciò che è stato preso prima del tempo
    # di start. L'opzione inplace fa si che non venga 
    # creata una copia del DataFrame originale.
    df.drop(df[df.iloc[:,0] < start_time].index, inplace=True)

def drop_time(df, col):
    '''
    La funzione riceve un DataFrame ed elimina la colonna passata
    come argomento.
    '''
    df.drop(columns=col, inplace=True)
    
def remove_offset(df, offset):
    '''
    La funzione riceve un DataFrame di Pandas e un numero reale,
    che rappresenta l'offset dei dati raccolti, e corregge 
    l'offset da cui sono affetti i dati.
    '''
    df.iloc[:,1] = df.iloc[:,1] - offset

def amplify_data(df, amp):
    '''
    La funzione riceve un DataFrame di Pandas e un numero reale, che 
    rappresenta il valore dell'amplificazione della tensione sulla
    resistenza, e amplifica i dati relativi alla tensione in ingresso
    di un fattore fisso di 7.4, mentre la colonna delle tensioni sulla
    resitenza vengono amplificate del fattore passato come argomento.
    '''
    # Dato che l'amplificazione sulla tensione in ingresso è fissa la 
    #inserisco direttamente nel codice, senza che venga passata dal 
    # chiamante. questo comportamento può essere facilmente cambiato.
    amp1 = 7.4
    df.iloc[:,0] = df.iloc[:,0] * amp1
    df.iloc[:,1] = df.iloc[:,1] * amp
    
def compute_data(df, resistance):
    '''
    La funzione calcola Vprobe e I (tensione di sonda e corrente 
    raccolta) partendo dalla tensione ingresso e tensione sulla 
    resistenza OPPORTUNAMENTE AMPLIFICATE.
    '''
    df.iloc[:,0] = df.iloc[:,0] - df.iloc[:,1]
    df.iloc[:,1] = df.iloc[:,1] / resistance

def sort_data (df, col):
    '''
    La funzione riceve un Dataframe d Pandas e riordina i dati secondo
    la colonna passata come argomento.
    '''
    df.sort_values(by=col, inplace=True)

def import_data(path, start_time, offset, amp, resistance):
    '''
    La funzione riceve il percorso del file con i dati da importare,
    un tempo di start da cui iniziare a considerare i dati, una 
    tensione di offset, un fattore di amplificazione e un valore 
    per la resistenza del circuito. La funzione importa i dati, 
    rimuove il tempo di latenza del generatore, l'offset e calcola 
    la tensione sulla sonda e la corrente raccolta da essa, infine 
    ordina il DataFrame secondo le tensioni di sonda in ordine 
    crescente.
    Restituisce un DataFrame di Pandas con due colonne:
        (Vp, I).
    '''
    ##############################################################
    # CODICE PROVVISORIO: Da definire nonappena si avrà a che fare
    #                     con dei dati "veri"
    ##############################################################
    # STRUTTURA DATI:
    # 0: time
    # 1: Vin
    # 2: Vr
    # 3: Vr2
    ##############################################################
    # with open(path, 'rb') as dtlg_file:
    #     data = dtlg.parse(dtlg_file).transpose()
    # df = pd.DataFrame(data=data)
    df = pd.read_csv(path, header = None, sep="\t")
    df.drop(df.iloc[:, 3:], inplace=True, axis=1)
    columnsTitles=[0,2,1]
    df=df.reindex(columns=columnsTitles)
    # df = df.rename(columns={3:2})
    #############################################################
    remove_starting(df, start_time)
    #elimino la colonna 0 che contiene gli indici di tempo.
    drop_time(df, 0)
    remove_offset(df, offset)
    amplify_data(df, amp)
    compute_data(df, resistance)
    # ordino i dati secondo l'ascissa, ovvero la colonna delle
    # tensioni di sonda (1).
    sort_data(df, 1)
    df.columns = ['Vp', 'I']
    
    return df