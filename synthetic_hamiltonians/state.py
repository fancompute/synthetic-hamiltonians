import numpy as np
import qutip as qt

from synthetic_hamiltonians import annihilator_for_site
from synthetic_hamiltonians.ponomarev import ponomarev_index, fock_to_ponomarev


def qt_fock_to_ponomarev(fock_state, N, M):
    '''Converts a qutip state or list of the form (m_1 m_2 m_3 ... m_M) into Ponomarev form'''

    # TODO finish this
    dim = ponomarev_index(N, M, exact_boson_number=False) + 1
    index = fock_to_ponomarev(fock_state, N, M)
    return qt.basis(dim, index)

    # data = np.zeros((dim, dim), dtype=complex)
    #
    # if type(fock_state) is qt.Qobj:
    #     pass


def qt_tensor_to_fock(tensor_state):
    '''Converts a many-body tensor state into a combination of up to D^N fock states.
    Example: [1 1 1 1] ==> 1/2(|11>+|10>+|01>+|00>)'''
    # TODO
    pass


def qt_ponomarev_to_fock(index, N, M):
    # TODO
    pass


# def create_initial_state_one_photon_per_bin(n_time_bins_and_photons):
#     '''Creates an initial state corresponding to n photons, one in each of d time bins'''
#     # TODO: Fock dimension cutoff
#     d = n_time_bins_and_photons
#     n = n_time_bins_and_photons
#     storage_state = [qt.create(n+1) * qt.basis(n+1, 0) for _ in range(d)]
#     register_state = [qt.basis(n+1, 0)]
#     return qt.tensor(storage_state + register_state)


def create_initial_state_all_photons_one_bin(num_sites, num_bosons,
                                             full_bin_index=0,
                                             use_ponomarev=False):
    '''Creates an initial state correspoding to n photons all contained in one time bin'''
    if use_ponomarev:
        fock_list = [0] * num_sites
        fock_list[full_bin_index] = num_bosons
        return qt_fock_to_ponomarev(fock_list, num_bosons, num_sites)
    else:
        fock = num_bosons + 1
        assert num_sites - 1 >= full_bin_index >= 0, "Time bin must be in range [0,D)"
        storage_state = [qt.basis(fock, 0) for _ in range(full_bin_index)] + \
                        [(qt.create(fock) ** num_bosons * qt.basis(fock, 0)).unit()] + \
                        [qt.basis(fock, 0) for _ in range(num_sites - full_bin_index - 1)]
        # if include_register:
        #     register_state = [qt.basis(fock, 0)]
        #     return qt.tensor(storage_state + register_state)
        # else:

        return qt.tensor(storage_state)


def get_photon_occupancies(state, num_sites=None, num_bosons=None, use_ponomarev=True):
    '''Returns a D+1 length list of photon number expectation values for each
    time bin plus the register (last element)'''
    # if use_ponomarev:
    #     # Get occupancies of the storage bins
    #     expectations = []
    #     for site in range(num_sites):
    #         a = annihilator_for_site(site=site, num_sites=num_sites, num_bosons=num_bosons,
    #         use_ponomarev=use_ponomarev)
    #         expectations.append(qt.expect(a.dag() * a, state))
    #     return expectations
    #
    # else:
    #     expectations = []
    #     N, D = get_n_d(state)
    #     # Get occupancies of the storage bins
    #     for i in range(D):
    #         a = get_bin_annihilator(D, N, time_bin_index=i)
    #         expectations.append(qt.expect(a.dag() * a, state))
    #     # Append the register expectations
    #     if not exclude_register:
    #         a = get_register_annihilator(D, N)
    #         expectations.append(qt.expect(a.dag() * a, state))
    #     return expectations

    # Get occupancies of the storage bins
    expectations = []
    for site in range(num_sites):
        a = annihilator_for_site(site=site, num_sites=num_sites, num_bosons=num_bosons, use_ponomarev=use_ponomarev)
        expectations.append(qt.expect(a.dag() * a, state))
    return expectations


def get_two_photon_correlations(state, num_sites=None, num_bosons=None, use_ponomarev=True):
    '''Computes the correlation <ai† aj† ai aj> for each lattice site i, j for a given state'''
    two_photon_correlations = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        ai = annihilator_for_site(i, num_sites=num_sites, num_bosons=num_bosons, use_ponomarev=use_ponomarev)
        for j in range(num_sites):
            aj = annihilator_for_site(j, num_sites=num_sites, num_bosons=num_bosons, use_ponomarev=use_ponomarev)
            two_photon_correlations[i, j] = qt.expect(ai.dag() * ai * aj.dag() * aj, state)
    return two_photon_correlations
