# Questo file raccoglie le fuznioni utili all fitting dei
# dati.

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import odr

def compute_V_floating(df):
    return df.iloc[:,0][df.iloc[:,1].abs().idxmin()]

def run_fit(data, model, parameters):
    '''
    La funzione riceve un dataframe di pandas formattato come (x, y, errx, erry), un
    modello di odr e una lista di parametri e fitta i dati con ODR. Restituisce il 
    risultato del fit.
    '''
    odrData = odr.RealData(data.iloc[:, 0],
                           data.iloc[:, 1],
                           sx=data.iloc[:, 2],
                           sy=data.iloc[:, 3])
    odr_fit = odr.ODR(odrData, model, beta0=parameters)
    return odr_fit.run()


def initial_fit(data, model, x_max, initial_parameters):
    '''
    La funzione avvia la prima iterazione del fit.
    '''
    # scelgo solo i valori che stiano nel intervallo in cui si vuole
    # eseguire il fit
    data_fit = data.drop(data[data.iloc[:, 0] > x_max].index)
    return run_fit(data_fit, model, initial_parameters)


def temp_plot(Te, n_iterations, best_iter, Te_best):
    '''
    La funzione costruisce il grafico della temperatura elettronica
    in funzione del numero di iterazioni.
    '''
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(1, n_iterations + 1, 1),
            Te, '.', label=r'$T_e$')
    ax.plot(best_iter + 1, Te_best, 'o', label=r'$T_e^{best}$')
    ax.set_xlabel(r'# iteration')
    ax.set_ylabel(r'$T_e$ [eV]')
    ax.grid(True, ls=':')
    ax.legend()
    return fig


def fit_figure(df, fit_function, fit_parameters, last_Vp):
    fig, ax = plt.subplots(1, 1)
    ax.errorbar((df.iloc[:, 0]).array, (df.iloc[:, 1]).array,
                xerr=(df.iloc[:, 2]).array, 
                yerr=(df.iloc[:, 3]).array,
                ls='', label='Real Data',
                fmt='o', markersize=2, elinewidth=.7, ecolor='k', capsize=1
                )
    fit_plot_x = np.linspace(np.min(df.iloc[:, 0]),
                             np.max(df.iloc[:, 0]), 1000)
    ax.plot(fit_plot_x, fit_function(fit_parameters, fit_plot_x), 'C1',
            label='fit result')
    ax.set_ybound(np.min(df.iloc[:, 1])
                  * 2, np.max((df.iloc[:, 1]))*1.2)
    ax.set_xlabel(r'$V_{probe}$ [V]')
    ax.set_ylabel(r'$I$ [mA]')
    ax.grid(ls=':')
    ax.axvline(x=last_Vp, ls='--', lw=.5, color='k')
    ax.legend()
    return fig


def fit_iteration(data, function, n_iterations, x_max_first_iter, step,
                  initial_parameters):
    '''
    La funzione rivece nell'ordine:
        - Un DataFrame di Pandas con formattazione del tipo (x, y, errx, erry)
        - Una funzione per eseguire il fit
        - Il numero di iterazioni da eseguire
        - L'estremo sinistro dell'area del fit
        - L'estremo destro per la prima iterazione
        - Lo step da aggiungere all'area di fit per ogni iterazione
        - Una lista con i valori iniziali dei parametri della funzione di fit

        La funzione richiede che venga caricato il modulo ODR di scipy

        Il metodo migliore per farsi ritornare tutti i dati potrebbe essere un 
        dizionario
    '''
    # Definisco la funzione modello per ODR
    model = odr.Model(function)
    # eseguo la prima iterazione del fit per avere un primo set di parametri
    # sottoforma di array di numpy. Tengo traccia del valore ottenuto per i parametri
    # ad ogni iterazione
    # con l'array fit_par
    # voglio un array con 4 colonne e per ora nessuna riga
    fit_par = np.zeros((0, len(initial_parameters)))
    fit_par_errors = np.zeros((0, len(initial_parameters)))
    fit_Chi2 = np.zeros(0)
    # impilo nell'array il risultato del primo fit e definisco anche il vettore Te
    # che terrà traccia delle temperature elettroniche ottenute per ogni iterazione
    # del fit
    fit_initial_out = initial_fit(
        data, model, x_max_first_iter, initial_parameters)

    fit_par = np.vstack((fit_par, fit_initial_out.beta))
    fit_par_errors = np.vstack((fit_par_errors, fit_initial_out.sd_beta))
    fit_Chi2 = np.append(fit_Chi2, fit_initial_out.res_var)

    Te = fit_par[0, 3]

    for i in range(1, n_iterations):
        # scelgo solo i valori che stiano nel intervallo in cui si vuole
        # eseguire il fit
        # notare che mi riferisco alle colonne del dataframe mediante la loro
        # posizione (iloc) e non mediante la loro etichetta, per guadagnare in
        # generalità.
        data_fit = data.drop(
            data[data.iloc[:, 0] > x_max_first_iter + i * step].index)
        # i-1: per inizializzare il fit devo prendere i parametri relativi al giro
        # precedente
        odr_out = run_fit(data_fit, model, fit_par[i-1, :])

        Te = np.append(Te, odr_out.beta[3])
        fit_Chi2 = np.append(fit_Chi2, odr_out.res_var)
        fit_par = np.vstack((fit_par, odr_out.beta))
        fit_par_errors = np.vstack((fit_par_errors, odr_out.sd_beta))

    # trovo la temperatura minima e mantengo il fit corrispondente
    best_iter = np.argmin(Te)
    Te_best = Te[best_iter]
    # fit_par_best = fit_par[best_iter, :]
    # fit_par_error_best = fit_par_errors[best_iter, :]
    # listed_fit_par_best = list(fit_par_best)
    # listed_fit_par_best_error = list(fit_par_error_best)
    # print(listed_fit_par_best)
    # print(listed_fit_par_best_error)
    zipped_par = zip(list(fit_par[best_iter, :]),
                     list(fit_par_errors[best_iter, :]))
    fit_par_best = list(zipped_par)
    fit_Chi2_best = fit_Chi2[best_iter]
    Vp_last = x_max_first_iter + best_iter * step
    # creo la figura con il plot della temperatura in funzione del numero di
    # iterazionimche poi verrà ritornata al completamento della funzione insieme agli
    # altri dati
    temp_fig = temp_plot(Te, n_iterations, best_iter, Te_best)

    # costriusco il grafico dei dati fittati sovrapposti alla curva di fit
    fit_fig = fit_figure(data, function, fit_par[best_iter, :], Vp_last)
    return {
        'last_Vp'           : Vp_last,
        'Te'                : Te_best,
        'fit_par'           : fit_par_best,
        #'fit_par_errors'    : fit_par_error_best,
        'Chi2'              : round(fit_Chi2_best, 2),
        'temperature_plot'  : temp_fig,
        'fit_plot'          : fit_fig,
        'best_iteration'    : best_iter + 1
    }