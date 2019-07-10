#!/usr/bin/env python
''' Funzione che ritorna un array degli indici di dove i dati non saturano'''
# -*- coding: utf-8 -*-

# Dichiaro le librerie
import numpy as np
#import matplotlib.pyplot as plt

def filtro(Vin, Vr):
    #devinisco il vettore sei separatori e il vettore vuoto da restituire
    sep = np.where(np.abs(Vin + 6.8) < 0.02)[0]
    indici = np.zeros(0)
    print(sep)
    print(np.max(Vr[sep[3]:sep[4]]))
    # Per ogni intervallo controlla se satura e se non lo fa aggiunge gli
    # indici all' array
    for i in range(len(sep)-1):
        if np.all(Vr[sep[i]:sep[i+1]] < 9.8):
            indici = np.append(indici, np.arange(sep[i], sep[i+1]))
            
    return indici

