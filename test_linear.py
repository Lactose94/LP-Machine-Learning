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

    norm_F_orig = np.linalg.norm(config.forces, axis=1)
    norm_F_fitted = np.linalg.norm(F_reg, axis=1)
    # calculate the mean of the relative deviation from original force
    delta_F = (F_reg - config.forces) / norm_F_orig.reshape(norm_F_orig.size, 1)
    F_mean = np.mean(np.linalg.norm(delta_F, axis=1))
    F_var = np.var(np.linalg.norm(delta_F, axis=1))

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
    return np.array([F_mean, F_var, delta_e])

def main():
    N = 10
    nqs = (6, 8, 10, 12, 14)
    steps = (150, 100, 50, 25, 1)
    with open('test_data/linear_quality.dat', 'w') as linear_out:
        for stepsize in steps:
            for modi in nqs:
                print(f'{stepsize}, {modi}')
                us_cfg = {
                        "file_in": "OUTCAR.21",
                        "file_out": "data",
                        "stepsize": stepsize,
                        "cutoff": 4,
                        "nr_modi": modi,
                        "lambda": 1e-12,
                        "kernel": ["linear",]
                    }
                with open('user_config.json', 'w') as json_out:
                    json.dump(us_cfg, json_out)
                os.system('python calibration.py')
                fitting = np.zeros(3)
                prediction = np.zeros(3)
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
                fit_str = '\t'.join(str(nr) for nr in fitting)
                pred_str = '\t'.join(str(nr) for nr in prediction)
                line = f'{stepsize}\t{modi}\t' + fit_str + '\t' + pred_str + '\n'
                linear_out.write(line)

if __name__ == '__main__':
    main()
