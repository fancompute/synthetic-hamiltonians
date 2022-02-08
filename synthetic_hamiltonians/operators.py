import numpy as np
import scipy
import matplotlib as mpl

from scipy.special import comb

from multiprocessing import Pool

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm.notebook import tqdm
from tqdm.contrib.concurrent import process_map as tqdm_parallel
from functools import reduce
from copy import copy
import qutip as qt

from synthetic_hamiltonians.ponomarev import *

_ANNIHILATOR_CACHE = {}


def _get_ponomarev_annihilator(site_index, total_sites, num_bosons):
    if (site_index, total_sites, num_bosons) in _ANNIHILATOR_CACHE:
        return _ANNIHILATOR_CACHE[(site_index, total_sites, num_bosons)]
    N = num_bosons  # N bosons
    M = total_sites  # M sites

    dim = ponomarev_index(N, M) + 1
    data = np.zeros((dim, dim), dtype=complex)

    for index in range(1, dim):
        fock = ponomarev_to_fock(index, N, M)
        n_level = fock[site_index]
        if n_level > 0:
            from_index = index
            to_fock = copy(fock)
            to_fock[site_index] -= 1
            to_index = fock_to_ponomarev(to_fock, N, M)
            data[to_index, from_index] = np.sqrt(n_level)

    op = qt.Qobj(data)
    _ANNIHILATOR_CACHE[(site_index, total_sites, num_bosons)] = op

    return op


def get_n_d(state):
    fock_cutoff = state.dims[0][0]
    N = fock_cutoff - 1  # number of photons
    time_bins_plus_register = len(state.dims[0])
    D = time_bins_plus_register - 1  # number of time bins in the storage ring
    return N, D


def annihilator_for_site(site_index, total_sites, fock, use_ponomarev=True):
    '''Returns the annihilator I ⊗ I ⊗ ... ⊗ a ⊗ ... ⊗ I ⊗ I for a given site'''
    assert total_sites - 1 >= site_index >= 0, "Site must be in range [0,total_sites)"
    if use_ponomarev:
        num_bosons = fock - 1
        return _get_ponomarev_annihilator(site_index, total_sites, num_bosons)
    else:
        return qt.tensor([qt.qeye(fock) for _ in range(site_index)] + [qt.destroy(fock)] + [qt.qeye(fock) for _ in
                                                                                            range(
                                                                                                total_sites - site_index - 1)])


def get_register_annihilator(n_time_bins, n_photons, fock_cutoff=None):
    '''Returns the register annihilation operator I ⊗ I ⊗ ... ⊗ I ⊗ a.dag()'''
    fock = n_photons + 1 if fock_cutoff is None else fock_cutoff
    return qt.tensor([qt.qeye(fock) for _ in range(n_time_bins)] + [qt.destroy(fock)])


def get_bin_annihilator(n_time_bins, n_photons, time_bin_index, include_register=True, fock_cutoff=None):
    '''Returns I ⊗ I ⊗ ... ⊗ a.dag() ⊗ ... ⊗ I ⊗ I'''
    assert time_bin_index <= n_time_bins - 1, "Time bin must be in range [0,D)"
    fock = n_photons + 1 if fock_cutoff is None else fock_cutoff
    storage_op = [qt.qeye(fock) for _ in range(time_bin_index)] + [qt.destroy(fock)] + [qt.qeye(fock) for _ in range(
        n_time_bins - time_bin_index - 1)]
    if include_register:
        register_op = [qt.qeye(fock)]
        return qt.tensor(storage_op + register_op)
    else:
        return qt.tensor(storage_op)
