import copy
import json
import os
import sys
import numpy as np
import kernel
import configuration
import calibration

def float_to_str(nr: float):
    return str(nr).replace('.', '')

def test_data(c_path, w_path, e_path, json_path, offset=1, printing=True):
    descriptors = np.loadtxt(c_path)
    weights = np.loadtxt(w_path)
    e_ave = np.loadtxt(e_path)[0]
    with open(json_path, 'r') as u_conf:
            user_config = json.load(u_conf)

    # make a list of the allowed qs
    qs = np.arange(1, user_config['nr_modi']+1) * np.pi / user_config['cutoff']

    # read in data and save parameters for calibration comparison
    (nc_new, ni_new, lat, configurations) = calibration.load_data(user_config, offset)

    config = configurations[0]

    if printing:
        for key, value in user_config.items():
            if type(value) != list:
                print(f'{key:>15}: {value:<15}')
            else:
                print(f'{key:>15}: {value[0]:<15}')
                print(f'{"sigma":>15}: {value[1]:<15}')

    nc_old = int(weights.size / 64)
    ni_old = 64

    kern = kernel.Kernel(*user_config['kernel'])

    config.init_nn(user_config['cutoff'], lat)
    config.init_descriptor(qs)

    E, F_reg = kern.predict(qs, config, descriptors, weights, e_ave)

    norm_F_orig = np.linalg.norm(config.forces, axis=1)
    norm_F_fitted = np.linalg.norm(F_reg, axis=1)
    # calculate the mean of the relative deviation from original force
    delta_F = (F_reg - config.forces) / norm_F_orig.reshape(norm_F_orig.size, 1)
    F_mean = np.mean(np.linalg.norm(delta_F, axis=1))
    F_var = np.var(np.linalg.norm(delta_F, axis=1))

    cos_forces = (F_reg * config.forces).sum(axis=1) / (norm_F_orig * norm_F_fitted)
    mean_cos_forces = np.mean(cos_forces)
    var_cos_forces = np.var(cos_forces)
    # relative energy difference
    delta_e = abs((E - config.energy) / config.energy)


    if printing:
        print('\nPredicted values:')
        print(f'energy = {E}')
        print(f'forces = {F_reg[1]}\n')

        print('Values from Outcar:')
        print(f'energy = {config.energy}')
        print(f'forces = {config.forces[1]}')


        print(f'Mean norm of difference:\n {F_mean} +- {F_var}')
        print(f'Relative to size of F:\n {F_mean / np.mean(np.linalg.norm(config.forces, axis=1))}')
    return np.array([F_mean, F_var, mean_cos_forces, var_cos_forces, delta_e])


def test_sigmas(n: int, min_sigma, max_sigma, modi):
    N = 10
    # FIXME: for small sigma the predicted force is always zero
    if min_sigma == 0:
        sigmas = np.linspace(1e-5, max_sigma, n)
    else:
        sigmas = np.linspace(min_sigma, max_sigma, n)
    meanf_fit = []
    varf_fit = []
    meancos_fit = []
    varcos_fit = []
    e_fit = []

    meanf_pred = []
    varf_pred = []
    meancos_pred = []
    varcos_pred = []
    e_pred = []
    for sigma in sigmas:
        us_cfg = {
            "file_in": "OUTCAR.21",
            "file_out": "data",
            "stepsize": 25,
            "cutoff": 4,
            "nr_modi": modi,
            "lambda": 1e-12,
            "kernel": ["gaussian", sigma]
        }
        with open('user_config.json', 'w') as json_out:
            json.dump(us_cfg, json_out)
        os.system('python calibration.py')
        fitting = np.zeros(5)
        prediction = np.zeros(5)
        step = int(round(1000 / us_cfg["stepsize"], 0))

        for i in range(N):
            print(0 + step * i, 1 + step * i)
            fitting += test_data('data/calibration_C.out',
                                'data/calibration_w.out',
                                'data/calibration_E.out',
                                'user_config.json',
                                0 + step * i,
                                False)
            prediction += test_data('data/calibration_C.out',
                                'data/calibration_w.out',
                                'data/calibration_E.out',
                                'user_config.json',
                                1 + step * i,
                                False)
        fitting = fitting / N
        prediction = prediction / N

        meanf_fit.append(fitting[0])
        varf_fit.append(fitting[1])
        meancos_fit.append(fitting[2])
        varcos_fit.append(fitting[3])
        e_fit.append(fitting[4])

        meanf_pred.append(prediction[0])
        varf_pred.append(prediction[1])
        meancos_pred.append(prediction[2])
        varcos_pred.append(prediction[3])
        e_pred.append(prediction[4])
        print(f'{round((sigma / sigmas[-1]), 2) * 100}%')

    stepsize = us_cfg['stepsize']
    # TODO: update such the nr of configs can also be iterated
    np.savetxt(f'test_data/fit_{n}_({float_to_str(min_sigma)}-{float_to_str(max_sigma)})_m{modi}_s{stepsize}.dat', np.array([sigmas, meanf_fit, varf_fit, meancos_fit, varcos_fit ,e_fit]).T)
    np.savetxt(f'test_data/prediction_{n}_({float_to_str(min_sigma)}-{float_to_str(max_sigma)})_m{modi}_s{stepsize}.dat', np.array([sigmas, meanf_pred, varf_pred, meancos_pred, varcos_pred, e_pred]).T)


def main():
    arg_in = sys.argv
    if len(arg_in) != 5:
        print('please supply nr of steps, min, max sigma and nr of modi')
        sys.exit(1)
    else:
        n = int(arg_in[1])
        min_sigma = float(arg_in[2])
        max_sigma = float(arg_in[3])
        modi = int(arg_in[4])
        test_sigmas(n, min_sigma, max_sigma, modi)

if __name__ == '__main__':
    main()
