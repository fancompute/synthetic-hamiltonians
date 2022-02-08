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

from synthetic_hamiltonians.ponomarev import ponomarev_index, fock_to_ponomarev


# def create_initial_state_one_photon_per_bin(n_time_bins_and_photons):
#     '''Creates an initial state corresponding to n photons, one in each of d time bins'''
#     # TODO: Fock dimension cutoff
#     d = n_time_bins_and_photons
#     n = n_time_bins_and_photons
#     storage_state = [qt.create(n+1) * qt.basis(n+1, 0) for _ in range(d)]
#     register_state = [qt.basis(n+1, 0)]
#     return qt.tensor(storage_state + register_state)


def qt_tensor_to_fock(tensor_state):
    '''Converts a many-body tensor state into a combination of up to D^N fock states.
    Example: [1 1 1 1] ==> 1/2(|11>+|10>+|01>+|00>)'''

    # TODO
    pass


def qt_fock_to_ponomarev(fock_state, N, M):
    '''Converts a qutip state or list of the form (m_1 m_2 m_3 ... m_M) into Ponomarev form'''

    # TODO finish this
    dim = ponomarev_index(N, M) + 1
    index = fock_to_ponomarev(fock_state, N, M)
    return qt.basis(dim, index)

    # data = np.zeros((dim, dim), dtype=complex)
    #
    # if type(fock_state) is qt.Qobj:
    #     pass


def qt_ponomarev_to_fock(index, N, M):

    # TODO

    pass


def create_initial_state_all_photons_one_bin(n_time_bins, n_photons, full_bin_index=0,
                                             fock_cutoff=None,
                                             include_register=True,
                                             use_ponomarev=False):
    '''Creates an initial state correspoding to n photons all contained in one time bin'''
    if use_ponomarev:
        fock_list = [0] * n_time_bins
        fock_list[full_bin_index] = n_photons
        return qt_fock_to_ponomarev(fock_list, n_photons, n_time_bins)
    else:
        d = n_time_bins
        n = n_photons
        if fock_cutoff is None:
            fock = n + 1
        else:
            fock = fock_cutoff
        assert n_time_bins - 1 >= full_bin_index >= 0, "Time bin must be in range [0,D)"
        storage_state = [qt.basis(fock, 0) for _ in range(full_bin_index)] + [
            (qt.create(fock) ** n * qt.basis(fock, 0)).unit()] + [qt.basis(fock, 0) for _ in
                                                                  range(n_time_bins - full_bin_index - 1)]
        if include_register:
            register_state = [qt.basis(fock, 0)]
            return qt.tensor(storage_state + register_state)
        else:
            return qt.tensor(storage_state)
