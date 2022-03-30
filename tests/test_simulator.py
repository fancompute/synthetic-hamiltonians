import unittest
import numpy as np
from itertools import product

from synthetic_hamiltonians.state import *
from synthetic_hamiltonians.bosehubbard import *
from synthetic_hamiltonians.simulator import *
from synthetic_hamiltonians.utils import tqdm

np.set_printoptions(precision=3, linewidth=300)


class SimulatorTest(unittest.TestCase):

    @staticmethod
    def _get_hamiltonian_and_propagator(N=2, D=4, κ=0.2, μ=1, U=1,
                                        hamiltonian_type="1d_line",
                                        toroidal=False, use_ponomarev=True):

        if hamiltonian_type == "1d_line":
            H = BHH_1d_line(D, num_bosons=N,
                            toroidal=toroidal,
                            κ=κ, μ=μ, U=U,
                            include_chemical_potential=(μ != 0.0),
                            include_onsite_interaction=(U != 0.0),
                            add_extraneous_register_node=True,
                            use_ponomarev=use_ponomarev,
                            display_progress=False)
            G = construct_BHH_propagator_1d_line(D, num_bosons=N,
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
                            add_extraneous_register_node=True,
                            use_ponomarev=use_ponomarev,
                            display_progress=False)
            G = construct_BHH_propagator_2d_grid(nx, ny, num_bosons=N,
                                                 toroidal=toroidal,
                                                 κ=κ, μ=μ, U=U,
                                                 include_chemical_potential=(μ != 0.0),
                                                 include_onsite_interaction=(U != 0.0),
                                                 use_ponomarev=use_ponomarev,
                                                 display_progress=False)
        else:
            raise NotImplementedError("TODO")

        return H, G

    def test_BHH_propagator(self, display_progress=True):
        '''Tests that the propagators are close to expected e^-iHt from exact BH model'''
        kappa_list = [0.001]
        mu_list = [0.0, 1.0]
        U_list = [1.0, 1.0, 9.85]
        toroidal_list = [False, True]
        hamiltonian_type_list = ["2d_grid", "1d_line"]
        N = 1
        D = 4

        all_combinations = list(product(*[kappa_list, mu_list, U_list, toroidal_list, hamiltonian_type_list]))
        if display_progress:
            all_combinations = tqdm(all_combinations)

        for κ, μ, U, toroidal, hamiltonian_type in all_combinations:
            # print(κ, μ, U, toroidal, hamiltonian_type)
            H, G = SimulatorTest._get_hamiltonian_and_propagator(N=N, D=D, κ=κ, μ=μ, U=U,
                                                                 toroidal=toroidal,
                                                                 hamiltonian_type=hamiltonian_type,
                                                                 use_ponomarev=True)

            iterations = 1000

            G_expected = (-1j * H * iterations).expm()
            G_expected_np = G_expected.full()

            G_actual_np = (G ** iterations).full()

            # print(G_actual_np)
            # print('\n\n')
            # print(G_expected_np)

            # For testing purposes we want to ignore: (1) the extraneous 0 index (2) the vacuum state index, and
            # (3) the phase of the register bin because nothing ends up in there at the end

            extraneous_index = 0
            vacuum_state_index = fock_to_ponomarev([0, 0, 0, 0, 0], N=N, M=D + 1, exact_boson_number=False)
            register_index = fock_to_ponomarev([0, 0, 0, 0, 1], N=N, M=D + 1, exact_boson_number=False)

            G_expected_np = np.delete(G_expected_np, [extraneous_index, vacuum_state_index, register_index], axis=0)
            G_expected_np = np.delete(G_expected_np, [extraneous_index, vacuum_state_index, register_index], axis=1)

            G_actual_np = np.delete(G_actual_np, [extraneous_index, vacuum_state_index, register_index], axis=0)
            G_actual_np = np.delete(G_actual_np, [extraneous_index, vacuum_state_index, register_index], axis=1)

            np.testing.assert_allclose(G_actual_np, G_expected_np, rtol=1e-02, atol=1e-06)


if __name__ == '__main__':
    unittest.main()
