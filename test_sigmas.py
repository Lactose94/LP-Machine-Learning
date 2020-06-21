import copy
import json
import os
import sys
import numpy as np
import kernel
import configuration
import calibration

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

    delta_F = F_reg - config.forces
    F_mean = np.mean(np.linalg.norm(delta_F, axis=1))
    F_var = np.var(np.linalg.norm(delta_F, axis=1))

    signs_reg = np.sign(F_reg)
    signs_ana = np.sign(config.forces)
    sign_diff = signs_reg - signs_ana

    if printing:
        print('\nPredicted values:')
        print(f'energy = {E}')
        print(f'forces = {F_reg[1]}\n')

        print('Values from Outcar:')
        print(f'energy = {config.energy}')
        print(f'forces = {config.forces[1]}')
        print('\nSign differences:', sign_diff[sign_diff != 0].size, 'out of', sign_diff.size)
        #print(sign_diff)


        print(f'Mean norm of difference:\n {F_mean} +- {F_var}')
        print(f'Relative to size of F:\n {F_mean / np.mean(np.linalg.norm(config.forces, axis=1))}')
    return(F_mean, F_var,  sign_diff[sign_diff != 0].size)


def test_sigmas(n: int, modi: int):
    sigmas = [i * .25 for i in range(1, n+1)]
    mean_fit = []
    var_fit = []
    signs_fit = []
    mean_pred = []
    var_pred = []
    signs_pred = []
    for sigma in sigmas:
        us_cfg = {
            "file_in": "OUTCAR.21",
            "file_out": "data",
            "stepsize": 100,
            "cutoff": 4,
            "nr_modi": modi,
            "lambda": 1e-12,
            "kernel": ["gaussian", sigma]
        }
        with open('user_config.json', 'w') as json_out:
            json.dump(us_cfg, json_out)
        os.system('python calibration.py')
        fitting = test_data('data/calibration_C.out',
                            'data/calibration_w.out',
                            'data/calibration_E.out',
                            'user_config.json',
                            0,
                            False)
        prediction = test_data('data/calibration_C.out',
                               'data/calibration_w.out',
                               'data/calibration_E.out',
                               'user_config.json',
                               1,
                               False)
        mean_fit.append(fitting[0])
        var_fit.append(fitting[1])
        signs_fit.append(fitting[2])

        mean_pred.append(prediction[0])
        var_pred.append(prediction[1])
        signs_pred.append(prediction[2])
        print(sigma)

    np.savetxt(f'test_data/fit_{n}-{modi}.dat', np.array([sigmas, mean_fit, var_fit, signs_fit]).T)
    np.savetxt(f'test_data/prediction_{n}-{modi}.dat', np.array([sigmas, mean_pred, var_pred, signs_pred]).T)


def main():
    arg_in = sys.argv
    if len(arg_in) != 3:
        print('please supply nr of 0.25 steps and nr of modis')
        sys.exit(1)
    else:
        n = int(arg_in[1])
        modi = int(arg_in[2])
        test_sigmas(n, modi)

if __name__ == '__main__':
    main()
