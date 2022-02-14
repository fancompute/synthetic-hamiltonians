import unittest

from synthetic_hamiltonians.ponomarev import *


class PonomarevTest(unittest.TestCase):

    # Check that these functions invert themselves
    @staticmethod
    def _fock_ponomarev_invertible(N=6, M=6, exact_boson_number=True):
        values_checked = 0
        for index in range(1, ponomarev_index(N, M, exact_boson_number=exact_boson_number) + 1):
            fock = ponomarev_to_fock(index, N, M, exact_boson_number=exact_boson_number)
            index_out = fock_to_ponomarev(fock, N, M, exact_boson_number=exact_boson_number)
            if index_out != index:
                raise ValueError(f"{ponomarev_to_fock(index, N, M, exact_boson_number=exact_boson_number)=}, but" +
                                 f" {fock_to_ponomarev(fock, N, M, exact_boson_number=exact_boson_number)=} !")
            else:
                values_checked += 1

        return values_checked

    # Check that these functions invert themselves
    def test_fock_ponomarev_inverse_exact_boson_number(self, max_N=6, max_M=6):
        values_checked_exact = 0
        values_checked_inexact = 0
        for N in range(1, max_N + 1):
            for M in range(1, max_M + 1):
                values_checked_exact += PonomarevTest._fock_ponomarev_invertible(N=N, M=M, exact_boson_number=True)
                values_checked_inexact += PonomarevTest._fock_ponomarev_invertible(N, M, exact_boson_number=False)
                self.assertGreater(values_checked_exact, 0)
                self.assertGreater(values_checked_inexact, 0)
        print(f"All {values_checked_exact} exact values are invertable for {max_N=}, {max_M=}.")
        print(f"All {values_checked_inexact} inexact values are invertable for {max_N=}, {max_M=}.")


if __name__ == '__main__':
    unittest.main()
