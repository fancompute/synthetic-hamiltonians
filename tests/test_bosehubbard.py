import unittest
import numpy as np
from itertools import product

from synthetic_hamiltonians.state import *
from synthetic_hamiltonians.bosehubbard import *
from synthetic_hamiltonians.utils import tqdm

np.set_printoptions(precision=3, linewidth=300)


class BoseHubbardTest(unittest.TestCase):

    @staticmethod
    def _evolve_BHH(N=2, D=4, κ=0.2, μ=1, U=1, tmax=25,
                    hamiltonian_type="1d_line",
                    toroidal=False, use_ponomarev=True,
                    initial_state="one_bin"):

        # Create initial state
        if initial_state == "one_bin":
            photon_location = D // 2
            state_initial = create_initial_state_all_photons_one_bin(num_sites=D, num_bosons=N,
                                                                     full_bin_index=photon_location,
                                                                     use_ponomarev=use_ponomarev)
        else:
            raise NotImplementedError("TODO")

        if hamiltonian_type == "1d_line":
            H = BHH_1d_line(D, num_bosons=N,
                            toroidal=toroidal,
                            κ=κ, μ=μ, U=U,
                            include_chemical_potential=(μ != 0.0),
                            include_onsite_interaction=(U != 0.0),
                            use_ponomarev=use_ponomarev,
                            display_progress=False)
        elif hamiltonian_type == "2d_grid":
            nx = int(np.sqrt(D))
            ny = int(np.sqrt(D))
            assert nx * ny == D
            H = BHH_2d_grid(nx, ny, num_bosons=N,
                            toroidal=toroidal,
                            κ=κ, μ=μ, U=U,
                            include_chemical_potential=(μ != 0.0),
                            include_onsite_interaction=(U != 0.0),
                            use_ponomarev=use_ponomarev,
                            display_progress=False)
        else:
            raise NotImplementedError("TODO")

        states_over_time = []

        for t in range(0, tmax):
            state_evolved = (-1j * H * t).expm() * state_initial
            states_over_time.append(state_evolved)

        photon_expectations_over_time = np.array([
            np.array(get_photon_occupancies(state, num_sites=D, num_bosons=N, use_ponomarev=use_ponomarev))
            for state in states_over_time
        ])
        two_photon_correlations_over_time = np.array([
            get_two_photon_correlations(state, num_sites=D, num_bosons=N, use_ponomarev=use_ponomarev)
            for state in states_over_time
        ])

        return states_over_time, photon_expectations_over_time, two_photon_correlations_over_time

    @staticmethod
    def _results_are_close(result1, result2, rtol=1e-05, atol=1e-08):
        '''Return whether two sim results have the same photon expectations and correlations'''
        _, expectations1, correlations1 = result1
        _, expectations2, correlations2 = result2
        return np.allclose(expectations1, expectations2, rtol=rtol, atol=atol) and \
               np.allclose(correlations1, correlations2, rtol=rtol, atol=atol)

    def test_BHH_ponomarev_equivalence(self, display_progress=True):
        '''Tests that states evolve to the same photon expectations and correlations under Ponomarev and standard
        state space representations.'''
        kappa_list = [0.1, 1.0, 9.23]
        mu_list = [0.0, 1.0]
        U_list = [1.0, 1.0, 9.85]
        toroidal_list = [False, True]
        hamiltonian_type_list = ["1d_line", "2d_grid"]

        tmax = 10

        all_combinations = list(product(*[kappa_list, mu_list, U_list, toroidal_list, hamiltonian_type_list]))
        if display_progress:
            all_combinations = tqdm(all_combinations)

        for κ, μ, U, toroidal, hamiltonian_type in all_combinations:
            res1 = BoseHubbardTest._evolve_BHH(N=1, D=4, κ=κ, μ=μ, U=U, tmax=tmax,
                                               toroidal=toroidal,
                                               hamiltonian_type=hamiltonian_type,
                                               use_ponomarev=False)
            res2 = BoseHubbardTest._evolve_BHH(N=1, D=4, κ=κ, μ=μ, U=U, tmax=tmax,
                                               toroidal=toroidal,
                                               hamiltonian_type=hamiltonian_type,
                                               use_ponomarev=True)
            equivalent = BoseHubbardTest._results_are_close(res1, res2)
            # print(f"{equivalent=}:     {κ=}, {μ=}, {U=}, {toroidal=}, {hamiltonian_type=}")

            self.assertTrue(equivalent)


if __name__ == '__main__':
    unittest.main()
