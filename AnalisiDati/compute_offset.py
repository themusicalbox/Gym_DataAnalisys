#!/usr/bin/env python3
'''Lo script calcola l'offset da usare per un set di dati nell'
analisi.'''

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libraries import dtlg
from libraries import filtering

path = sys.argv[1]

with open(path, 'rb') as dtlg_file:
    data = dtlg.parse(dtlg_file).transpose()
df = pd.DataFrame(data=data)

filtering.remove_starting(df, 1e-3)
df.columns = ['time', 'Vin', 'Vr']

plt.plot(df.time, df.Vin, lw=.3)
plt.plot(df.time, df.Vr, lw=.3)
plt.show()
print("OFFSET = ", np.mean(df.iloc[:, 2]))
