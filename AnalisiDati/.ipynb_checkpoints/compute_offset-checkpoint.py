#!/usr/bin/env python3

import sys 

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from libraries import dtlg
from libraries import filtering
# from libraries import binning
# from libraries import fitting
from scipy import odr

path = sys.argv[1]

with open(path, 'rb') as dtlg_file:
        data = dtlg.parse(dtlg_file).transpose()
df = pd.DataFrame(data=data)

filtering.remove_starting(df, 1e-3)

print("OFFSET = ", np.mean(df.iloc[:,2]))