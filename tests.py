import unittest
from numpy import array, array_equal
from outcar_parser import Parser


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
        self.assertIsInstance(nr_ions, int)
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

        wrong_parser = Parser('wrong_data_outcar.21')

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            wrong_parser.find_lattice_vectors()
            output = out.getvalue().strip()
            self.assertEqual(output, warn_message)
        finally:
            sys.stdout = saved_stdout

    def test_read_pos_force_energy(self):
        test_in = 'OUTCAR.21'

        g_pos = array([2.26725, 2.36995, 0.06367])
        g_force = array([-0.171492, -0.290427, -1.773642])
        g_energy = -306.41169589

        parser = Parser(test_in)
        self.assertIsNotNone(parser.build_configurations(1000))

        nr_of_configs = len(list(parser.build_configurations(1000)))
        self.assertEqual(nr_of_configs, 1)
        nr_of_configs = len(list(parser.build_configurations(1)))
        self.assertEqual(nr_of_configs, 1000)

        for energy, position, force in parser.build_configurations(1000):
            self.assertTrue(array_equal(position[0], g_pos))
            self.assertTrue(array_equal(force[0], g_force))
            self.assertEqual(energy, g_energy)

class TestCalibration(unittest.TestCase):

    # Test if the q-vector is build correctly
    def test_build_q(self):
        pass
    
    # Test if the program panics if the cutoff is bigger than a/2
    def test_cutoff_to_big(self):
        pass


class TestKernel(unittest.TestCase):
    
    # Test if choosing the kernel works and panics of no sigma is given
    def test_kernel_choice(self):
        pass
    
    # Tests if the shape and value of the kernel-fcts is the expected
    def test_kernel_value(self):
        pass
    
    # tests if the shape and value of the matrix element is the expected
    def test_matrix_element(self):