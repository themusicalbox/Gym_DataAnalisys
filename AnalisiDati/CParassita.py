#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:15:11 2019

@author: fabio
"""
import numpy as np
import matplotlib.pyplot as plt
from libraries import dtlg


# Lettura Dati
with open('../dati/190503/190503003', 'rb') as data:
    time, Vin, Vr = dtlg.parse(data)

# Derivata Vin
dVin = np.gradient(Vin, (time[2]-time[1]), edge_order=2)
#plt.plot(dVin[20:2000], '.', markersize=1)

# Calcolo Cp
# calcolo la corrente nel circuito
I = Vr / 270
# tolgo gli elementi con dVin == 0)
I = I[np.where(dVin != 0)]
dVin = dVin[np.where(dVin != 0)]
# calcolo i valori di C a ogni istante
allC = I / dVin
allC = allC[300:]
plt.plot(allC, '.', markersize=0.5)
# calcolo la media (dovrebbe essere ~1000 pF per 10m di cavo)
Cp = np.nanmean(allC)
print(Cp)
print("\n<<<CAPACITÀ PARASSITA>>>\n",
      "Cp = %i" %(Cp/1e-12), " [pF]")

plt.show()
