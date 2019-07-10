#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dichiaro le librerie
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from scipy import odr
from scipy.optimize import curve_fit
from libraries import dtlg as dtlg

# Parametri della sonda
L = 0.01125
d = 0.002
A = L * d * np.pi
R = 997.4

# Costanti fisiche
e = 1.6e-19
k = 1.38e-23
me = 9e-31

# Parameti tecnici
start_time = 5e-4  # tempo prima che il generatore vada a regime
offset = .05  # offset dell'op amp
amp1 = 1  # amplificazione della tensione in uscita
amp2 = 7.4  # amplificazione della tensione in entrata

# Imposto la renderizzazione dei grafici con latex
#matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = [
#    r'\usepackage[eulermath]{classicthesis}',
#    r'\usepackage{siunitx}'
#    r'\usepackage{amssymb}'
#    r'\newcommand{\vett}[1]{\mathbf{#1}}'
#    r'\renewcommand{\vec}{\vett}']

########################################
# DEFINIZIONE DI FUNZIONI
########################################
# le definizioni di funzioni vanno spostate in
# in un file a parte
###


def langmuir(par, x):
    return par[0] * (1 - par[1] * (x - par[2]) -
                     np.exp((x - par[2]) / par[3]))

################################################################################


# Lettura Dati
with open('dati/180412003', 'rb') as data:
    time, Vin, Col2, Vr = dtlg.parse(data)
# questo con un po' di lavoro si può fare più semplicemente con un dataset di pandas

# elimino i dati presi prima che il generatore sia
# andato a regime.
index = np.flatnonzero(time > start_time)
time = time[index]
Vin = Vin[index]
Vr = Vr[index]

# correggo l'offset dei valori
Vr = Vr - offset

# Trovo Vprobe
Vp = Vin * amp2 - Vr * amp1

# Calcolo la corrente in mA
I = Vr * amp1/R * 1000

# ordino l'array con Vp mantenendo la corrispondenza
# con le correnti
index = np.argsort(Vp, axis=-1)
Vp = Vp[index]
I = I[index]

# BINNING DEI DATI
# definisco due intervalli su cui binnare i dati:
# uno per la regione ad andamento lineare, uno per
# la regione ad andamento esponenziale

# bisogna definire quale sia il binning ideale
# per ora venogono inseriti alcuni valori di test
n_bins_linear = 100
n_bins_exp = 400
Va_linear = np.linspace(np.min(Vp), -1,
                        num=n_bins_linear)
Va_exp = np.linspace(-1, np.max(Vp),
                     num=n_bins_exp)
# si uniscono i due intervalli in un unico vettore
Va = np.append(Va_linear, Va_exp)

Vp_binned = np.zeros(0)
I_binned = np.zeros(0)
Vp_error = np.zeros(0)
I_error = np.zeros(0)

# si procede a binnare i dati mediando il contenuto
# di ciascun bin.
for i in range(len(Va)-1):
    cond1 = np.flatnonzero(Vp < Va[i+1])
    cond2 = np.flatnonzero(Vp > Va[i])
    cond = np.intersect1d(cond1, cond2)
    if cond.size != 0:
        Vp_binned = np.append(Vp_binned, np.nanmean(Vp[cond]))
        I_binned = np.append(I_binned, np.nanmean(I[cond]))
        Vp_error = np.append(Vp_error, np.std(Vp[cond]))
        I_error = np.append(I_error, np.std(I[cond]))

plt.plot(Vp, I, '.', markersize=.1, label=r'Raw Data')
plt.errorbar(Vp_binned, I_binned, xerr=Vp_error, yerr=I_error, ls='',
             label=r'Binned', fmt='oC2', markersize=2, elinewidth=.7, ecolor='k', capsize=1)

# Trovo Vfloat
Iabs = np.absolute(I_binned)
Vfl = Vp_binned[np.argmin(Iabs)]
print("V_float = ", Vfl)

###########################################
# FITTING DEI DATI
###########################################

# Trovo Temperatura minima
### I = Isat [1 - aplha(Vp - Vfl) - exp(e/k * (Vp - Vfl) / Te)]

model = odr.Model(langmuir)
par = [0.001, 0.0001, -1, 1]
step = 1

par3 = np.zeros(0)
iterations = np.zeros(0)

for i in range(30):
    index_fit = np.flatnonzero(Vp_binned < (Vfl + i * step))
    Vp_fit = Vp_binned[index_fit]
    I_fit = I_binned[index_fit]
    errx_fit = Vp_error[index_fit]
    erry_fit = I_error[index_fit]
    data = odr.RealData(Vp_fit, I_fit, sx=errx_fit, sy=erry_fit)
    odr_fit = odr.ODR(data, model, beta0=par)
    output = odr_fit.run()
    fit_par = output.beta
    par3 = np.append(par3, fit_par[3])
    # TODO creare vettore errori su par3
    iterations = np.append(iterations, i)
# output.pprint()


# y_fit = langmuir(fit_par, Vp_binned)
# plt.plot(Vp_binned, y_fit)
# plt.grid(ls=':')
# lgnd = plt.legend(markerscale=1)

ax = plt.gca()
ax.set_ylim([-0.2, 2.5])
#ax.set_xlabel(r'$V_{\text{probe}} \, [\si{\volt}]$')
#ax.set_ylabel(r'$I \, [\si{\milli \ampere}]$')

fig = plt.figure()
Tele = par3 * (e / k)
plt.plot(iterations, par3, '.')

# Fit della Temperatura Elettronica per trovarne il minimo
# def funcT (par, x):
#     return np.poly1d([par[0],par[1],par[2]], variable = x)
# dataT = odr.Data(par3) #TODO aggiungere errori par3
# modelT = odr.Model(funcT)
# odr_fitT = odr.ODR(dataT, modelT, beta0=[1,1,1])
# outputT = odr_fitT.run()
# approxT = np.poly1d(output.beta)
# #crit_points =  #TODO trovare minimo di approxT

# TODO Re-fit dei dati con la Temperatura elettronica trovata

plt.show()
