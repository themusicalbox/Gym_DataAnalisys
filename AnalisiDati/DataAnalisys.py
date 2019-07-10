#!/usr/bin/env python
''' Il programma esegue l'analisi dei dati per l'eseprimento sulle sonde assiali
sulla macchina lineare Gym'''
# -*- coding: utf-8 -*-

# Dichiaro le librerie
import os
import shutil
import argparse
import time
import json
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from libraries import filtering
from libraries import binning
from libraries import fitting


# TODO: Calcoare capacità parassita


def langmuir(par, x):
    '''
    La fuznione definisce la curva di langmuir
    riceve una variabile e quattro parametri.
    '''
    return par[0] * (1 - par[1] * (x - par[2]) -
                     np.exp((x - par[2]) / par[3]))


def cli_args():
    '''
    Command line argument parser
    '''
    parser = argparse.ArgumentParser(
        description='Run the analisys on dataset series of Gym data.')
    parser.add_argument('input', metavar='JSON',
                        type=argparse.FileType(),
                        help=('config file in JSON format'))
    parser.add_argument('-sf', '--save-figures', dest='save_figures',
                        action='store_const',
                        const=True, default=False,
                        help='Saves figures on disk (default: False), slower')
    parser.add_argument('-j', '--parallel', dest='j', metavar='JOBS',
                        action='store',
                        type=int,
                        default=1,
                        help=('The maximum number of concurrent processes\
                             to use, default 1'))
    return parser.parse_args()


def compute_probe_area(probe_parameters):
    '''La funzione riceve i parametri della sonda (lunghezza e sezione) e ne
    calcola l'area di raccolta.'''
    L = probe_parameters["L"]
    d = probe_parameters["d"]
    # TODO: Stimare l'errore sull'area della sonda
    return (L * d * np.pi, 0)


def plot_raw_data(df, path):
    '''
    La funzione plotta e salva i dati grezzi
    '''
    fig, ax = plt.subplots(1, 1)
    ax.plot(df['Vp'], df['I'], '.', markersize=.1)
    ax.set_xlabel(r'$V_{probe}$ [V]')
    ax.set_ylabel(r'$I$ [A]')
    ax.grid(ls=':')
    fig.set_size_inches(8, 6)
    fig.savefig(path + "/raw_data.png")


def save_figures(df, fit_out, path):
    '''
    Questa funzione viene chiamata per salvare le figure delle varie elaborazioni
    '''
    plot_raw_data(df, path)
    fit_out["temperature_plot"].savefig(path + "/temp_plot.png")
    fit_out["fit_plot"].savefig(path + "/fit_plot.png")


def global_log_string(global_data_settings, probe_parameters):
    '''
    La funzione crea la sringa con i parametri globali
    dell'acquisizione
    '''
    return (
        '{:*^50}\n'.format('GyM Data Analisys') +
        'Analisys started on ' +
        time.strftime('%d %b %Y %H:%M:%S %Z') + '\n\n' +
        'ACQUISITION PARAMETERS:\n' +
        '{0:<20} = {1[amplification]:<13}{2}\n'.format(
            'Amplification', global_data_settings, '[-]') +
        '{0:<20} = {1[offset]:<13.4f}{2}\n'.format(
            'Offset', global_data_settings, '[V]') +
        '{0:<20} = {1[tension_interval][0]:<+13.0f}{2}\n'.format(
            'V_in min', global_data_settings, '[V]') +
        '{0:<20} = {1[tension_interval][1]:<+13.0f}{2}\n'.format(
            'V_in max', global_data_settings, '[V]') +
        '{0:<20} = {1[resistance]:<13}{2}\n'.format(
            'Resistance', global_data_settings, '[Ω]') +
        '{0:<20} = {1[MW_source]!r:<13}{2}\n\n'.format(
            'Microwave source', global_data_settings, '[-]') +
        'PROBE PARAMETERS:\n' +
        '{0:<20} = {1[L]:<13.5f}{2}\n'.format(
            'Lenght',
            probe_parameters, '[m]') +
        '{0:<20} = {1[L]:<13.3f}{2}\n'.format(
            'Diameter',
            probe_parameters, '[m]')
    )


def elaborate_data(dataset):
    '''
    La funzione lancia l'elaborazione su un dataset
    '''
    in_file = in_dir + dataset["id"]
    print("reading : ", in_file)
    # setting output path
    out_path = config["output_dir"] + dataset["id"]

    # import and format data
    data = filtering.import_data(in_file, data_settings,
                                 ramp_filter=dataset["ramp_filter"])

    # build zones list
    if zones[0] == "min":
        zones[0] = np.min(data.Vp)
    if zones[-1] == "max":
        zones[-1] = np.max(data.Vp)

    # bin data
    binned_data = binning.bin_data(data, zones, bins)
    # fit data
    fit_output = fitting.fit_iteration(data=binned_data,
                                       function=langmuir,
                                       n_iterations=fit_params["n_iter"],
                                       step=fit_params["step"],
                                       in_file=dataset["id"])

    plasma_par = fitting.compute_plasma_parameters(fit_output["fit_par"],
                                                   probe_area)

    # if the save_figure flag is up, save figures on disk
    if args.save_figures:
        os.makedirs(out_path)
        save_figures(data, fit_output, out_path)

    # build dataset dictionary of data to be returned
    plt.close('all')
    ret_data = {
        "id": dataset["id"],
        "probe_position": dataset["pos"],
        "B": dataset["B"],
        "pressure": dataset["pressure"],
        "power": dataset["power"],
        "MW_source": config["data_settings"]["MW_source"],
        "Te": plasma_par["T_e"][0],
        "errTe": plasma_par["T_e"][1],
        "Vplasma": plasma_par["V_plasma"][0],
        "errVplasma": plasma_par["V_plasma"][1],
        "Vfloat": plasma_par["V_fl"][0],
        "errVfloat": plasma_par["V_fl"][1],
        "ne": plasma_par["n_e"][0],
        "err_ne": plasma_par["n_e"][1]
    }
    log_results = (
        '\n{:-^40}\n'.format('-') +
        '{:<20}{!r}\n\n'.format('File:', dataset["id"]) +
        '{:<20}{:<13}{:>6}\n'.format('Probe position:', dataset["pos"], '[mm]') +
        '{:<20}{:<13}{:>6}\n'.format('B:', dataset["B"], '[ A]') +
        '{:<20}{:<13}{:>6}\n'.format('Pressure:', dataset["pressure"], '[mbar]') +
        '{:<20}{:<13.0f}{:>6}\n\n'.format('Power:', dataset["power"]*100, '[ %]') +
        'FITTING RESULTS:\n' +
        '{:<25} = {:<7.2f}{}\n'.format(
            'V_fl from plot', fit_output["init_V_fl"], '[ V]') +
        '{:<25} = {:<7}{}\n'.format(
            'number of interations', fit_params["n_iter"], '[--]') +
        '{:<25} = {:<7}{}\n'.format('step', fit_params["step"], '[--]') +
        '{:<25} = {:<7}{}\n'.format(
            '# best iteration', fit_output["best_iteration"], '[--]') +
        '{:<25} = {:<7.2f}{}\n'.format(
            'V_p best iteration', fit_output["last_Vp"], '[ V]') +
        '\nParameters:\n' +
        '{0:<10} = {1[0]:>+9.5f} ± {1[1]:<10.5f}{2}\n'.format(
            'I_i,sat', fit_output["fit_par"][0], '[ A]') +
        '{0:<10} = {1[0]:>+9.5f} ± {1[1]:<10.5f}{2}\n'.format(
            'α', fit_output["fit_par"][1], '[--]') +
        '{0:<10} = {1[0]:>+9.5f} ± {1[1]:<10.5f}{2}\n'.format(
            'V_float', fit_output["fit_par"][2], '[ V]') +
        '{0:<10} = {1[0]:>+9.5f} ± {1[1]:<10.5f}{2}\n'.format(
            'T_e', fit_output["fit_par"][1], '[eV]') +
        '{:-^40}\n'.format('-')
    )
    return (ret_data, log_results)


if __name__ == "__main__":
    # start timestamp to compute execution time
    t1 = time.time()
    # parse cli args
    args = cli_args()
    # load config file
    config = json.load(args.input)

    # parse config
    in_dir = config["input_dir"]
    data_settings = config["data_settings"]
    zones = config["binning"]["zones"]
    bins = config["binning"]["bins"]
    probe_area = compute_probe_area(config["probe_parameters"])
    fit_params = config["fitting"]

    # defining log destination
    log_file_name = "fit_results.log"
    log_file_path = config['output_dir'] + log_file_name

    # create first log string
    log_string = global_log_string(config["data_settings"],
                                   config["probe_parameters"])
    print('\n\n' + log_string)

    # create output dir, delete if already exists
    if os.path.isdir(config["output_dir"]):
        shutil.rmtree(config["output_dir"])
        print("existing output dir deleted\n")
    os.makedirs(config["output_dir"])

    # write log string on log file
    log_file = open(log_file_path, 'a')
    log_file.write(log_string)

    # write fitting heading on log file
    log_file.write('\n\n{:-^40}\n'.format('Fitting data'))
    log_file.close()

    # initializing results table
    result_summary = pd.DataFrame(
        columns=["id", "probe_position", "B", "pressure", "power", "MW_source", "Te",
                 "errTe", "Vplasma", "errVplasma", "Vfloat", "errVfloat",
                 "ne", "err_ne"]
    )

    # MULTIPROCESSING
    # creating processes
    pool = multiprocessing.Pool(args.j)
    # launch processes
    print("Elaboration launched on ", args.j, " jobs\n")
    results_list = pool.map(elaborate_data, config["datasets"])
    # post elaboration: print and save results
    print("\nElaboration done.\n")
    results_list = list(zip(*results_list))
    result_summary = result_summary.append(list(results_list[0]))
    print(result_summary)

    save_results_path = config["output_dir"] + "results_summary.json"
    result_summary.to_json(save_results_path, orient='table', index=False)

    log_file = open(log_file_path, 'a')
    log_file.write(''.join(list(results_list[1])))
    log_file.close()

    # stop time to compute execution time
    t2 = time.time()
    print("\nElaboration result saved in: " +
          save_results_path + "\n" +
          "Fit log file saved in:  " +
          log_file_path + '\n'
          "\nEXECUTION TIME: %.2f [s]" % (t2 - t1))
