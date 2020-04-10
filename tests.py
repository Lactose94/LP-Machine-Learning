import unittest
from math import exp, sqrt
from numpy import array, array_equal, ones, zeros
from outcar_parser import Parser
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
        test_lattice = array([
            [10.546640000, 0.000000000, 0.000000000],
            [0.000000000, 10.546640000, 0.000000000],
            [0.000000000, 0.000000000, 10.546640000]
            ])

        self.assertTrue(
            array_equal(parser.find_lattice_vectors(), test_lattice), 'lattice vectors do not match'
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

        g_pos = array([2.26725, 2.36995, 0.06367])
        g_force = array([-0.171492, -0.290427, -1.773642])
        g_energy = -306.41169589

        parser = Parser(test_in)
        self.assertIsNotNone(parser.build_configurations(1000), 'does not build iterator')

        nr_of_configs = len(list(parser.build_configurations(1000)))
        self.assertEqual(nr_of_configs, 1, 'does not return at least one config')
        nr_of_configs = len(list(parser.build_configurations(1)))
        self.assertEqual(nr_of_configs, 1000, 'does not return the correct number of configs')

        for energy, position, force in parser.build_configurations(1000):
            self.assertTrue(array_equal(position[0], g_pos), 'returns wrong position')
            self.assertTrue(array_equal(force[0], g_force), 'returns wrong force')
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
        self.assertEqual(kernel.linear_kernel(array(1), array(0)), 0)
        self.assertEqual(kernel.linear_kernel(ones(10), ones(10)), 10)

        self.assertEqual(kernel.gaussian_kernel(zeros(10), zeros(10), 10), 1)
        self.assertAlmostEqual(kernel.gaussian_kernel(array([sqrt(2)*3, 0]), zeros(2), 3), exp(1), 7)

    # Test if choosing the kernel works and panics if no sigma is given
    def test_kernel_choice(self):
        kern = kernel.Kernel('linear')
        self.assertEqual(kern.kernel.__name__, 'linear_kernel', 'does not set linear kernel')
        with self.assertRaises(ValueError):
            kernel.Kernel('gaussian')

    # tests if the shape and value of the matrix element is the expected
    def test_matrix_element(self):
        pass

    # tests if the shape and value of the subrow is correct
    def test_subrow(self):
        pass
