import unittest
from math import exp, sqrt
import numpy as np
from outcar_parser import Parser
from configuration import Configuration
import kernel

class TestParser(unittest.TestCase):

    def test_file_opening(self):
        correct_path = 'OUTCAR.21'

        correct_parser = Parser(correct_path)
        self.assertEqual(correct_parser.filepath, correct_path)
        self.assertIsNotNone(correct_parser.outcar_content)

    def test_file_not_opening(self):
        wrong_path = 'mydata.21'

        with self.assertRaises(ValueError):
            Parser(wrong_path)

    def test_ion_nrs(self):
        test_in = 'OUTCAR.21'
        parser = Parser(test_in)
        nr_ions = parser.find_ion_nr()
        self.assertIsInstance(nr_ions, int, 'ions should have type int')
        self.assertEqual(nr_ions, 64, f'nr of ions should be 64, is {nr_ions}')

    def test_read_lattice_vecors(self):
        test_in = 'OUTCAR.21'
        parser = Parser(test_in)
        test_lattice = np.array([
            [10.546640000, 0.000000000, 0.000000000],
            [0.000000000, 10.546640000, 0.000000000],
            [0.000000000, 0.000000000, 10.546640000]
            ])

        self.assertTrue(
            np.array_equal(parser.find_lattice_vectors(), test_lattice), 'lattice vectors do not match'
            )

    def test_warn_not_cubic(self):
        import sys
        from io import StringIO

        warn_message = '*************WARNING*************\nThe given lattice vectors\n' \
                       '[[10.       0.       0.     ]\n [ 0.      10.54664  0.     ]\n' \
                       ' [ 0.       0.      10.54664]]\n' \
                       'do not constitute a simple basic lattice.\n' \
                       'The programm wont work correctly'

        wrong_parser = Parser('test_data/wrong_data_outcar.21')

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            wrong_parser.find_lattice_vectors()
            output = out.getvalue().strip()
            self.assertEqual(output, warn_message, 'Warning message is not correct')
        finally:
            sys.stdout = saved_stdout

    def test_read_pos_force_energy(self):
        test_in = 'OUTCAR.21'

        g_pos = np.array([2.26725, 2.36995, 0.06367])
        g_force = np.array([-0.171492, -0.290427, -1.773642])
        g_energy = -306.41169589

        parser = Parser(test_in)
        self.assertIsNotNone(parser.build_configurations(1000), 'does not build iterator')

        nr_of_configs = len(list(parser.build_configurations(1000)))
        self.assertEqual(nr_of_configs, 1, 'does not return at least one config')
        nr_of_configs = len(list(parser.build_configurations(1)))
        self.assertEqual(nr_of_configs, 1000, 'does not return the correct number of configs')

        for energy, position, force in parser.build_configurations(1000):
            self.assertTrue(np.array_equal(position[0], g_pos), 'returns wrong position')
            self.assertTrue(np.array_equal(force[0], g_force), 'returns wrong force')
            self.assertEqual(energy, g_energy, 'returns wrong energy')


class TestCalibration(unittest.TestCase):

    # Test if the q-vector is build correctly
    def test_build_q(self):
        pass

    # Test if the program panics if the cutoff is bigger than a/2
    def test_cutoff_too_big(self):
        pass


class TestKernel(unittest.TestCase):
    # Tests if the shape and value of the kernel-fcts is the expected
    def test_kernel_values(self):
        self.assertEqual(kernel.linear_kernel(np.array([1, 0]), np.array([0, 0])), 0)
        self.assertEqual(kernel.linear_kernel(np.ones(10), np.ones(10)), 10)

        desc = np.zeros(2).reshape(2, 1)
        self.assertEqual(kernel.gaussian_kernel(desc, desc, 10)[0, 0], 1)
        desc1 = np.array([sqrt(2)*3, 0]).reshape(2, 1)
        desc2 = np.zeros(2).reshape(2, 1)
        self.assertAlmostEqual(kernel.gaussian_kernel(desc1, desc2, 3)[0, 0], exp(-1), 7)

    # Test if choosing the kernel works and panics if no sigma is given
    def test_kernel_choice(self):
        kern = kernel.Kernel('linear')
        self.assertEqual(kern.kernel_mat.__name__, 'linear_kernel', 'does not set linear kernel')
        with self.assertRaises(ValueError):
            kernel.Kernel('gaussian')

    # tests if the value of the matrix element is the expected
    def test_linear_energy_matrix_element(self):
        kern = kernel.Kernel('linear')
        descr1 = np.eye(10, 10)
        zero_el = np.sum(kern.kernel_mat(descr1, np.zeros(10)), axis=0)
        self.assertEqual(type(zero_el), np.float64)
        self.assertEqual(zero_el, 0)

        one = np.array([1] + [0 for _ in range(9)])
        one_el = np.sum(kern.kernel_mat(descr1, one), axis=0)
        self.assertEqual(one_el, 1)

        one = np.ones(10)
        ten_el = np.sum(kern.kernel_mat(descr1, one), axis=0)
        self.assertEqual(ten_el, 10)

        five = 5 * one
        fifty_el = np.sum(kern.kernel_mat(descr1, five), axis=0)
        self.assertEqual(fifty_el, 50)

    # tests the correct shape of the energy matrix in the linear case
    def test_linear_energy_matrix_shape(self):
        desc1 = np.ones((50, 5))
        desc2 = np.ones((400, 5))

        shape = np.shape(kernel.linear_kernel(desc1, desc2))
        self.assertEqual(shape, (50, 400))

    # tests the correct shape of the energy matrix in the gaussian case
    def test_gaussian_energy_matrix_shape(self):
        desc1 = np.ones((50, 5))
        desc2 = np.ones((400, 5))

        shape = np.shape(kernel.gaussian_kernel(desc1, desc2, 1))
        self.assertEqual(shape, (50, 400))

    # tests the values of the linear energy matrix for a simple example
    def test_linear_energy_matrix_values(self):
        kern = kernel.Kernel('linear')
        expected_val = np.eye(10)

        lin_eye = kern.kernel_mat(np.eye(10), np.eye(10))
        self.assertTrue(np.array_equal(lin_eye, expected_val))

    # test the values of the gaussian energy matrix for a simple example
    def test_gaussian_energy_matrix_values(self):
        kern = kernel.Kernel('gaussian', 1)
        expected_val = np.ones((10, 10)) * exp(-1)
        np.fill_diagonal(expected_val, 1)

        exp_eye = kern.kernel_mat(np.eye(10), np.eye(10))
        self.assertTrue(np.array_equal(exp_eye, expected_val))

    # tests if the shape and value of the subrow is correct
    def test_linear_energy_subrow(self):
        descr1 = np.eye(20, 10)
        descr2 = np.zeros((20, 10))
        descr2[0, 0] = 1

        kern = kernel.Kernel('linear')
        subrow = np.sum(kern.kernel_mat(descr1, descr2), axis=1)
        self.assertEqual(np.shape(subrow), (20, ))
        self.assertEqual(subrow[0], 1)
        self.assertTrue(np.array_equal(subrow[1:], np.zeros(19)))

        descr2 = np.eye(20, 10)

        subrow = np.sum(kern.kernel_mat(descr1, descr2), axis=1)
        self.assertTrue(np.array_equal(subrow[:10], np.ones(10)))
        self.assertTrue(np.array_equal(subrow[10:], np.zeros(10)))

    # tests if the linear kernel is self consistent
    def test_linear_cosistency(self):
        import json
        import copy

        import numpy as np
        import kernel
        import configuration
        import calibration

        # load teset data
        descriptors = np.loadtxt('test_data/c_lin_10.out')
        weights = np.loadtxt('test_data/w_lin_10.out')
        with open('test_data/lin_10.json', 'r') as u_conf:
            user_config = json.load(u_conf)

        # make a list of the allowed qs
        qs = np.arange(1, user_config['nr_modi']+1) * np.pi / user_config['cutoff']

        # read in data and save parameters for calibration comparison
        (_, _, lat, configurations) = calibration.load_data(user_config)
        config = configurations[0]

        # load linear kernel
        kern = kernel.Kernel(*user_config['kernel'])

        # init descriptors of the test configuration
        config.init_nn(user_config['cutoff'], lat)
        config.init_descriptor(qs)

        # make perturbed configs and init them
        dx = 1e-4
        config_plus = copy.deepcopy(config)
        config_minus = copy.deepcopy(config)

        config_plus.positions[0, 0] += dx
        config_minus.positions[0, 0] -= dx

        config_plus.init_nn(user_config['cutoff'], lat)
        config_plus.init_descriptor(qs)

        config_minus.init_nn(user_config['cutoff'], lat)
        config_minus.init_descriptor(qs)

        # build the perturbed matrix elements
        kplus = np.sum(kern.kernel_mat(config_plus.descriptors, descriptors), axis=0)
        kminus = np.sum(kern.kernel_mat(config_minus.descriptors, descriptors), axis=0)

        # make the finite difference values
        eplus = kplus @ weights
        eminus = kminus @ weights

        # calculate forces by finite differences
        Fx_finite = (eplus - eminus) / (2 * dx)

        # calculate forces by regression
        F_reg = kern.force_submat(qs, config, descriptors) @ weights

        print('\n', Fx_finite - F_reg[0])
        self.assertAlmostEqual(Fx_finite, F_reg[0], 6)

    def test_gaussian_consistency(self):
        import json
        import copy

        import numpy as np
        import kernel
        import configuration
        import calibration

        # load teset data
        descriptors = np.loadtxt('test_data/c_gaus_10.out')
        weights = np.loadtxt('test_data/w_gaus_10.out')
        with open('test_data/gaus_10.json', 'r') as u_conf:
            user_config = json.load(u_conf)

        # make a list of the allowed qs
        qs = np.arange(1, user_config['nr_modi']+1) * np.pi / user_config['cutoff']

        # read in data and save parameters for calibration comparison
        (_, _, lat, configurations) = calibration.load_data(user_config)
        config = configurations[0]

        # load linear kernel
        kern = kernel.Kernel(*user_config['kernel'])

        # init descriptors of the test configuration
        config.init_nn(user_config['cutoff'], lat)
        config.init_descriptor(qs)

        # make perturbed configs and init them
        dx = 1e-4
        config_plus = copy.deepcopy(config)
        config_minus = copy.deepcopy(config)

        config_plus.positions[0, 0] += dx
        config_minus.positions[0, 0] -= dx

        config_plus.init_nn(user_config['cutoff'], lat)
        config_plus.init_descriptor(qs)

        config_minus.init_nn(user_config['cutoff'], lat)
        config_minus.init_descriptor(qs)

        # build the perturbed matrix elements
        kplus = np.sum(kern.kernel_mat(config_plus.descriptors, descriptors), axis=0)
        kminus = np.sum(kern.kernel_mat(config_minus.descriptors, descriptors), axis=0)

        # make the finite difference values
        eplus = kplus @ weights
        eminus = kminus @ weights

        # calculate forces by finite differences
        Fx_finite = (eplus - eminus) / (2 * dx)

        # calculate forces by regression
        F_reg = kern.force_submat(qs, config, descriptors) @ weights
        print('\n', Fx_finite - F_reg[0])
        self.assertAlmostEqual(Fx_finite, F_reg[0], 6)