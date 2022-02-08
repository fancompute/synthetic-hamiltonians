import unittest

import numpy as np

from synthetic_hamiltonians.ponomarev import *


class PonomarevTest(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    # Check that these functions invert themselves
    def test_fock_ponomarev_inverse(self, max_N=6, max_M=6):
        values_checked = 0
        for N in range(1, max_N + 1):
            for M in range(1, max_M + 1):
                for index in range(1, ponomarev_index(N, M) + 1):
                    fock = ponomarev_to_fock(index, N, M)
                    self.assertEqual(fock_to_ponomarev(fock, N, M), index,
                                     f"{ponomarev_to_fock(index, N, M)=}, but {fock_to_ponomarev(fock, N, M)=} !")
                    if fock_to_ponomarev(fock, N, M) != index:
                        raise ValueError(f"{ponomarev_to_fock(index, N, M)=}, but {fock_to_ponomarev(fock, N, M)=} !")
                    else:
                        values_checked += 1
        print(f"All {values_checked} values are invertable for {max_N=}, {max_M=}.")


if __name__ == '__main__':
    unittest.main()
