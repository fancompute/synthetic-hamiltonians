import unittest

import qutip as qt

from synthetic_hamiltonians.operators import *


class OperatorTest(unittest.TestCase):

    def test_ponomarev_annihilator(self, max_total_sites=4, max_num_bosons=4):

        values_checked = 0

        for N in range(max_num_bosons + 1):
            for M in range(max_total_sites + 1):

                dim = ponomarev_index(N, M) + 1

                for index in range(1, dim):
                    fock = ponomarev_to_fock(index, N, M)

                    for site_index in range(M):
                        n_level = fock[site_index]

                        qt_fock = qt.basis([N + 1] * M, fock)
                        qt_a = annihilator_for_site(site_index, M, N + 1, use_ponomarev=False)
                        qt_fock_new = qt_a * qt_fock

                        qt_pn = qt.basis(dim, index)
                        qt_pn_a = annihilator_for_site(site_index, M, N + 1, use_ponomarev=True)
                        qt_pn_new = qt_pn_a * qt_pn

                        # Check that amplitude is as expected
                        self.assertEqual(qt_pn_new.norm(), np.sqrt(n_level))
                        # Check that fock has reduced a state
                        fock_new = copy(fock)
                        fock_new[site_index] -= 1
                        qt_pn_new_expected = qt.basis(dim, fock_to_ponomarev(fock_new, N, M))
                        if qt_pn_new.norm() != 0:
                            self.assertEqual(qt_pn_new.unit(), qt_pn_new_expected.unit())
                        # Check that a.dag puts it back in original state
                        self.assertEqual(qt_pn_a.dag() * qt_pn_new, n_level * qt_pn)
                        values_checked += 1

        print(f"Annihilators behave correctly for all {values_checked} test cases" +
              f"for {max_total_sites=}, {max_num_bosons=}.")


if __name__ == '__main__':
    unittest.main()
