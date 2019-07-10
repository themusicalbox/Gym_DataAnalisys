'''
Il modulo raccoglie le funzioni utili a fare il fit sui dati.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
from scipy import constants as const


def compute_V_floating(df):
    '''
    prova
    '''
    return df.iloc[:, 0][df.iloc[:, 1].abs().idxmin()]


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
    fit_out = run_fit(data_fit, model, initial_parameters)
    return fit_out


def temp_plot(Te, n_iterations, best_iter, Te_best):
    '''
    La funzione costruisce il grafico della temperatura elettronica
    in funzione del numero di iterazioni.
    '''
    fig, axes = plt.subplots(1, 1)
    axes.plot(np.arange(1, n_iterations + 1, 1),
              Te, '.', label=r'$T_e$')
    axes.plot(best_iter + 1, Te_best, 'o', label=r'$T_e^{best}$')
    axes.set_xlabel(r'# iteration')
    axes.set_ylabel(r'$T_e$ [eV]')
    axes.grid(True, ls=':')
    axes.legend()
    return fig


def fit_figure(df, in_file, fit_function, fit_par, fit_par_errors, Chi2, last_Vp):
    '''La funzione construisce il grafico della corrente in funzione del
    potenziale di sonda sovrapposto alla curva di fit.'''
    fig, ax = plt.subplots(1, 1)
    ax.errorbar((df.iloc[:, 0]).array, (df.iloc[:, 1]).array,
                xerr=(df.iloc[:, 2]).array,
                yerr=(df.iloc[:, 3]).array,
                ls='', label='Real Data',
                fmt='o', markersize=2, elinewidth=.7, ecolor='k', capsize=1
                )
    fit_plot_x = np.linspace(np.min(df.iloc[:, 0]),
                             np.max(df.iloc[:, 0]), 1000)
    ax.plot(fit_plot_x, fit_function(fit_par, fit_plot_x), 'C1',
            label='fit result')
    ax.set_ybound(np.min(df.iloc[:, 1])
                  * 2, np.max((df.iloc[:, 1]))*1.2)
    ax.set_xlabel(r'$V_{probe}$ [V]')
    ax.set_ylabel(r'$I$ [mA]')
    ax.grid(ls=':')
    ax.axvline(x=last_Vp, ls='--', lw=.5, color='k')

    plot_text = (
        'Minimum temperature fitting: \n\n' +
        'id : ' + in_file + '\n\n'
        'last $V_{probe} = %.3f$ [V] \n' +
        '$ I_{i}^{sat} = %.5f \\pm %.5f$ [A]\n' +
        '$\\alpha = %.5f \\pm %.5f $\n' +
        '$V_{fl} = %.3f \\pm %.3f$ [V]\n' +
        '$T_e = %.3f \\pm %.3f $ [eV]\n' +
        "$\\chi^2 = %.3f $") % (last_Vp,
                                fit_par[0], fit_par_errors[0],
                                fit_par[1], fit_par_errors[1],
                                fit_par[2], fit_par_errors[2],
                                fit_par[3], fit_par_errors[3],
                                Chi2)
    ax.text(-47, .004, plot_text,
            bbox=dict(facecolor='white', edgecolor='black', ))
    ax.legend()
    return fig


def compute_electron_density(probe_area, fit_parameters):
    '''La funzione calcola, ricevendo parametri di fit e area di raccolta, la
    densità elettronica del plasma.'''
    # definisco tutte le grandezze in gioco come tuple del tipo (x, errx)
    # massa del deuterio
    md = (const.physical_constants["deuteron mass"][0],
          const.physical_constants["deuteron mass"][2])
    # carica dell'elettrone
    e = (const.physical_constants["elementary charge"][0],
         const.physical_constants["elementary charge"][2])
    # corrente di saturazione ionica
    I = fit_parameters[0]
    # temperatura elettronica
    Te = fit_parameters[3]
    n_e = (I[0]/(.6 * (-1) * e[0] * probe_area[0]) *
           np.sqrt(md[0] / (Te[0] * e[0])))

    I_contrib = 1/(e[0]**(1.5) * probe_area[0]) * np.sqrt(md[0]/Te[0]) * I[1]
    e_contrib = 1.5 * I[0]/probe_area[0] * np.sqrt(md[0]/Te[0]) * e[0]**(-2.5) *\
        e[1]
    A_contrib = I[0]/(probe_area[0]**2 * e[0]**1.5) * np.sqrt(md[0]/Te[0]) *\
        probe_area[1]
    m_contrib = .5 * I[0]/(e[0]**1.5 * probe_area[0] *
                           np.sqrt(Te[0] * md[0])) * md[1]
    T_contrib = .5 * I[0]/(e[0]**1.5 * probe_area[0]) * md[0]**.5 * Te[0]**(-1.5) *\
        Te[1]

    err_ne = np.sqrt(I_contrib**2 + e_contrib**2 + A_contrib**2 + m_contrib**2 +
                     T_contrib**2)
    return(n_e, err_ne)


def compute_V_plasma(fit_parameters):
    '''La funzione calcola, ricevendo i parametri del fit, il potenziale
    di plasma.'''
    Vfl = fit_parameters[2]
    Te = fit_parameters[3]
    return(Vfl[0] + 3.3 * Te[0],
           np.sqrt(Vfl[1]**2 + (3.3 * Te[1])**2))


def fit_iteration(data, in_file, function, n_iterations, step):
    '''
    La funzione rivece nell'ordine:
        - Un DataFrame di Pandas con formattazione del tipo (x, y, errx, erry)
        - Una funzione per eseguire il fit
        - Il numero di iterazioni da eseguire
        - L'estremo sinistro dell'area del fit
        - L'estremo destro per la prima iterazione
        - Lo step da aggiungere all'area di fit per ogni iterazione
        - Una lista con i valori iniziali dei parametri della funzione di fit

        Ritorna i dati del fit sottoforma di un dizionario
    '''
    # stimo Vfloating dal grafico dei dati
    Vfloat = compute_V_floating(data)
    # Definisco la funzione modello per ODR
    model = odr.Model(function)

    # parametri iniziali assegnati arbitrariamente
    initial_parameters = [.0001, .01, Vfloat, 1]

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

    # primo fit.
    fit_initial_out = initial_fit(
        data, model, Vfloat, initial_parameters)

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
            data[data.iloc[:, 0] > Vfloat + i * step].index)
        # data_fit = data.drop(
        #    data[data.iloc[:, 0] < Vfloat].index)
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
    fit_par_best = fit_par[best_iter, :]
    fit_par_best_errors = fit_par_errors[best_iter, :]
    fit_Chi2_best = fit_Chi2[best_iter]
    Vp_last = Vfloat + best_iter * step
    # creo la figura con il plot della temperatura in funzione del numero di
    # iterazionimche poi verrà ritornata al completamento della funzione insieme agli
    # altri dati
    temp_fig = temp_plot(Te, n_iterations, best_iter, Te_best)

    # costriusco il grafico dei dati fittati sovrapposti alla curva di fit
    fit_fig = fit_figure(data, in_file, function, fit_par_best, fit_par_best_errors,
                         fit_Chi2_best, Vp_last)

    fit_par_best = list(
        zip(list(fit_par_best),
            list(fit_par_best_errors)))
    return {
        'init_V_fl': Vfloat,
        'last_Vp': Vp_last,
        'fit_par': fit_par_best,
        'Chi2': round(fit_Chi2_best, 2),
        'temperature_plot': temp_fig,
        'fit_plot': fit_fig,
        'best_iteration': best_iter + 1
    }


def compute_plasma_parameters(fit_output_par, probe_area):
    '''
    La funzione riceve i parametri da un fit sottoforma di lista di tuple (x, errx) e
    calcola i parametri di plasma restituendoli in un dizionario.
    '''
    # costruisco un dizionario dei parametri di plasma
    return {
        'T_e': fit_output_par[3],
        'V_fl': fit_output_par[2],
        'n_e': compute_electron_density(probe_area, fit_output_par),
        'V_plasma': compute_V_plasma(fit_output_par)
    }


def print_results(plasma_parameters):
    '''La funzione stampa a nello standard output i parametri di plasma.'''
    n_e = plasma_parameters["n_e"]
    T_e = plasma_parameters["T_e"]
    V_fl = plasma_parameters["V_fl"]
    V_pl = plasma_parameters["V_plasma"]

    print("\n<<<<PARAMETRI DEL PLASMA>>>>")
    print(" n_e = %.2E ± %.2E m^-3\n" % (n_e[0], n_e[1]),
          "T_e = %.3f ± %.3f eV\n" % (T_e[0], T_e[1]),
          "V_fl = %.3f ± %.3f V\n" % (V_fl[0], V_fl[1]),
          "V_plasma =  %.3f ± %.3f V" % (V_pl[0], V_pl[1]))
