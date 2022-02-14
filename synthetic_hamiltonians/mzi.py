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

from synthetic_hamiltonians.operators import annihilator_for_site


def mzi_register(θ, ϕ, site, num_bosons=None, num_sites=None, use_ponomarev=False, register_index=-1):
    '''Operation corresponding to passing a time bin through the register MZI'''
    # fock = n + 1
    # a1 = get_register_annihilator(d, n, fock_cutoff=fock)  # register annihilator
    # a2 = get_bin_annihilator(d, n, time_bin_index=time_bin, fock_cutoff=fock)  # bin annihilator
    register_site = register_index % num_sites
    a1 = annihilator_for_site(register_site, num_sites, num_bosons, use_ponomarev=use_ponomarev)
    a2 = annihilator_for_site(site, num_sites, num_bosons, use_ponomarev=use_ponomarev)
    return (1j * θ / 2 * (np.exp(-1j * ϕ) * a1.dag() * a2 + np.exp(1j * ϕ) * a2.dag() * a1)).expm()


def swap_bin_into_register(site, num_bosons, num_sites, use_ponomarev=False):
    '''Swaps a photon pulse from a storage bin into the register, maintaining the phase'''
    return mzi_register(np.pi, np.pi / 2, site=site, num_bosons=num_bosons, num_sites=num_sites,
                        use_ponomarev=use_ponomarev)


def swap_bin_from_register(site, num_bosons, num_sites, use_ponomarev=False):
    '''Swaps a photon pulse from the register into a storage bin, maintaining the phase'''
    return mzi_register(np.pi, -np.pi / 2, site=site, num_bosons=num_bosons, num_sites=num_sites,
                        use_ponomarev=use_ponomarev)


def interact_register_bin(time_bin, κ, α, num_bosons, num_sites, use_ponomarev=False):
    '''Applies exp(i κ(e^-iα a1+ a2 + e^iα a2+ a1)) between the register and a time bin'''
    θ = κ * 2
    ϕ = α
    return mzi_register(θ, ϕ, time_bin, num_bosons=num_bosons, num_sites=num_sites,
                        use_ponomarev=use_ponomarev)


def interact_bin_bin(bin1, bin2, κ, α, num_bosons, num_sites, use_ponomarev=False):
    '''Transfers a bin to the register, applies exp(i κ(e^-iα a1+ a2 + e^iα a2+ a1))
    between the register and the time bin, and then returns the first bin to its original space'''
    θ = κ * 2
    ϕ = α
    return swap_bin_from_register(bin1, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev) * \
           mzi_register(θ, ϕ, bin2, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev) * \
           swap_bin_into_register(bin1, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev)
