import unittest

import qutip as qt

from functools import reduce

from synthetic_hamiltonians.operators import *


class OperatorTest(unittest.TestCase):

    def test_ponomarev_annihilator(self, max_total_sites=5, max_num_bosons=5):

        values_checked = 0

        for N in range(1, max_num_bosons + 1):
            for M in range(1, max_total_sites + 1):

                dim = ponomarev_index(N, M, exact_boson_number=False) + 1

                for index in range(1, dim):
                    fock_list = ponomarev_to_fock(index, N, M)

                    for site in range(M):
                        n_level = fock_list[site]

                        qt_fock = qt.basis([N + 1] * M, fock_list)
                        qt_a = annihilator_for_site(site, num_sites=M, num_bosons=N, use_ponomarev=False)
                        qt_fock_new = qt_a * qt_fock

                        qt_pn = qt.basis(dim, index)
                        qt_pn_a = annihilator_for_site(site, num_sites=M, num_bosons=N, use_ponomarev=True)
                        qt_pn_new = qt_pn_a * qt_pn

                        # Check that amplitude is as expected
                        self.assertEqual(qt_pn_new.norm(), np.sqrt(n_level))

                        # Check that the state reduces down to vacuum and back up
                        for level in range(1, n_level):
                            # Check that fock has reduced a state
                            fock_list_new = copy(fock_list)
                            fock_list_new[site] -= level
                            amplitude = reduce(lambda a, b: a * b, [np.sqrt(n_level - i) for i in range(level)])
                            qt_pn_new = (qt_pn_a ** level) * qt_pn
                            qt_pn_new_expected = amplitude * qt.basis(dim, fock_to_ponomarev(fock_list_new, N, M))
                            if level > n_level:  # annihilated the vacuum state
                                self.assertEqual(qt_pn_new.norm(), 0.0)
                            else:
                                self.assertEqual(qt_pn_new, qt_pn_new_expected)

                            # Check that a.dag puts it back in original state
                            self.assertEqual((qt_pn_a.dag() ** level) * qt_pn_new, (amplitude**2) * qt_pn)

                        values_checked += 1

        print(f"Annihilators behave correctly for all {values_checked} test cases " +
              f"for {max_total_sites=}, {max_num_bosons=}.")


if __name__ == '__main__':
    unittest.main()
