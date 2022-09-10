import numpy as np
from copy import copy
import qutip as qt

from synthetic_hamiltonians.ponomarev import *

_ANNIHILATOR_CACHE = {}


def _get_ponomarev_annihilator(site, num_sites, max_num_bosons):
    if (site, num_sites, max_num_bosons) in _ANNIHILATOR_CACHE:
        return _ANNIHILATOR_CACHE[(site, num_sites, max_num_bosons)]

    dim = ponomarev_index(max_num_bosons, num_sites, exact_boson_number=False) + 1
    data = np.zeros((dim, dim), dtype=complex)

    for index in range(1, dim):
        fock = ponomarev_to_fock(index, max_num_bosons, num_sites)
        n_level = fock[site]
        if n_level > 0:
            from_index = index
            to_fock = copy(fock)
            to_fock[site] -= 1
            to_index = fock_to_ponomarev(to_fock, max_num_bosons, num_sites)
            data[to_index, from_index] = np.sqrt(n_level)

    op = qt.Qobj(data)
    _ANNIHILATOR_CACHE[(site, num_sites, max_num_bosons)] = op

    return op


# def get_n_d(state):
#     fock_cutoff = state.dims[0][0]
#     N = fock_cutoff - 1  # number of photons
#     time_bins_plus_register = len(state.dims[0])
#     D = time_bins_plus_register - 1  # number of time bins in the storage ring
#     return N, D


def annihilator_for_site(site, num_sites, num_bosons, use_ponomarev=True):
    '''Returns the annihilator I ⊗ I ⊗ ... ⊗ a ⊗ ... ⊗ I ⊗ I for a given site'''
    assert num_sites - 1 >= site >= 0, "Site must be in range [0,num_sites)"
    if use_ponomarev:
        return _get_ponomarev_annihilator(site, num_sites, num_bosons)
    else:
        fock = num_bosons + 1
        return qt.tensor([qt.qeye(fock) for _ in range(site)] + [qt.destroy(fock)] +
                         [qt.qeye(fock) for _ in range(num_sites - site - 1)])


# def get_register_annihilator(n_time_bins, n_photons, fock_cutoff=None):
#     '''Returns the register annihilation operator I ⊗ I ⊗ ... ⊗ I ⊗ a.dag()'''
#     fock = n_photons + 1 if fock_cutoff is None else fock_cutoff
#     return qt.tensor([qt.qeye(fock) for _ in range(n_time_bins)] + [qt.destroy(fock)])
#
#
# def get_bin_annihilator(n_time_bins, n_photons, time_bin_index, include_register=True, fock_cutoff=None):
#     '''Returns I ⊗ I ⊗ ... ⊗ a.dag() ⊗ ... ⊗ I ⊗ I'''
#     assert time_bin_index <= n_time_bins - 1, "Time bin must be in range [0,D)"
#     fock = n_photons + 1 if fock_cutoff is None else fock_cutoff
#     storage_op = [qt.qeye(fock) for _ in range(time_bin_index)] + [qt.destroy(fock)] + [qt.qeye(fock) for _ in range(
#         n_time_bins - time_bin_index - 1)]
#     if include_register:
#         register_op = [qt.qeye(fock)]
#         return qt.tensor(storage_op + register_op)
#     else:
#         return qt.tensor(storage_op)
