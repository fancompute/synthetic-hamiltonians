import numpy as np
import scipy
import matplotlib as mpl

from scipy.special import comb

from multiprocessing import Pool

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functools import reduce
from copy import copy
import qutip as qt

from synthetic_hamiltonians.mzi import interact_bin_bin
from synthetic_hamiltonians.operators import annihilator_for_site
from synthetic_hamiltonians.ponomarev import ponomarev_index
from synthetic_hamiltonians.utils import tqdm


def construct_BHH_propagator_2d_grid(nx, ny, toroidal=False,
                                     n=None, d=None,
                                     κ=0.1, α=0,
                                     fock=None,
                                     μ=None, U=None,
                                     include_coupling=True,
                                     include_chemical_potential=True,
                                     include_onsite_interaction=True,
                                     use_ponomarev=False):
    num_nodes = nx * ny
    assert num_nodes == d
    nodes = list(range(num_nodes))
    nodes_xy = np.array(nodes).reshape((ny, nx))
    operations = []
    for y in range(ny):
        for x in range(nx):
            if x < nx - 1 or toroidal:  # we're not at the right edge
                operations.append(
                    interact_bin_bin(bin1=nodes_xy[y, x], bin2=nodes_xy[y, (x + 1) % nx], κ=κ, α=α, n=n, d=d))
            if y < ny - 1 or toroidal:  # we're not at the bottom edge
                operations.append(
                    interact_bin_bin(bin1=nodes_xy[(y + 1) % ny, x], bin2=nodes_xy[y, x], κ=κ, α=α, n=n, d=d))

    # Compute the propagator for one iteration
    return reduce(lambda U1, U2: U2 * U1, operations)  # left-multiplication


def make_BH_Hamiltonian(nodes, edges, fock=None, μ=None, U=None,
                        include_coupling=True,
                        include_chemical_potential=False,
                        include_onsite_interaction=True,
                        use_ponomarev=False,
                        display_progress=False):
    '''
    Returns the actual Bose-Hubbard Hamiltonian for a list of lattice sites and a list of couplings between sites
    Args:
        nodes: a list of node indices, e.g. [1,2,3,4]
        edges: a list of couplings between edges with coupling constants and phases, e.g. [((1,2), 0.1, np.pi), ((1,3), 0.15, np.pi/2), ...]
    '''
    if use_ponomarev:
        dim = ponomarev_index(fock - 1, len(nodes)) + 1
        H = qt.qzero(dim)
    else:
        H = qt.tensor([qt.qzero(fock) for _ in nodes])

    if include_coupling:
        iterator = tqdm if display_progress else lambda x: x
        for ((node1, node2), κ, α) in iterator(edges):
            a1 = annihilator_for_site(site_index=node1, total_sites=len(nodes), fock=fock, use_ponomarev=use_ponomarev)
            a2 = annihilator_for_site(site_index=node2, total_sites=len(nodes), fock=fock, use_ponomarev=use_ponomarev)
            # a1 = get_annihilator_index(index=node1, num_operators=len(nodes), fock=fock)
            # a2 = get_annihilator_index(index=node2, num_operators=len(nodes), fock=fock)
            H += -1 * κ * (np.exp(-1j * α) * a1.dag() * a2 + np.exp(1j * α) * a2.dag() * a1)

    if include_chemical_potential:
        for node in nodes:
            a = annihilator_for_site(site_index=node, total_sites=len(nodes), fock=fock, use_ponomarev=use_ponomarev)
            # a = get_annihilator_index(index=node, num_operators=len(nodes), fock=fock)
            H += -1 * μ * a.dag() * a

    if include_onsite_interaction:
        for node in nodes:
            a = annihilator_for_site(site_index=node, total_sites=len(nodes), fock=fock, use_ponomarev=use_ponomarev)
            # a = get_annihilator_index(index=node, num_operators=len(nodes), fock=fock)
            H += -1 * U * a.dag() * a.dag() * a * a

    return H


def BHH_2d_grid(nx, ny, toroidal=False,
                κ=0.1,
                fock=None,
                μ=None, U=None,
                include_coupling=True,
                include_chemical_potential=True,
                include_onsite_interaction=True,
                use_ponomarev=False,
                display_progress=False):
    num_nodes = nx * ny
    nodes = list(range(num_nodes))
    nodes_xy = np.array(nodes).reshape((ny, nx))
    edges = []
    for y in range(ny):
        for x in range(nx):
            if x < nx - 1:  # we're not at the right edge
                edges.append(((nodes_xy[y, x], nodes_xy[y, x + 1]), κ, 0))
            elif toroidal:
                edges.append(((nodes_xy[y, x], nodes_xy[y, (x + 1) % nx]), κ, 0))
            if y < ny - 1:  # we're not at the bottom edge
                edges.append(((nodes_xy[y, x], nodes_xy[y + 1, x]), κ, 0))
            elif toroidal:
                edges.append(((nodes_xy[y, x], nodes_xy[(y + 1) % ny, x]), κ, 0))
    return make_BH_Hamiltonian(nodes, edges, fock=fock, μ=μ, U=U,
                               include_coupling=include_coupling,
                               include_chemical_potential=include_chemical_potential,
                               include_onsite_interaction=include_onsite_interaction,
                               use_ponomarev=use_ponomarev,
                               display_progress=display_progress)


def BHH_1d_line(num_nodes, toroidal=False,
                κ=0.1,
                fock=None,
                μ=None, U=None,
                include_coupling=True,
                include_chemical_potential=True,
                include_onsite_interaction=True,
                use_ponomarev=False,
                display_progress=False):
    nodes = list(range(num_nodes))
    edges = []
    for i in range(num_nodes):
        if i < num_nodes - 1 or toroidal:  # we're not at the right edge
            edges.append(((nodes[i], nodes[(i + 1) % num_nodes]), κ, 0))
    return make_BH_Hamiltonian(nodes, edges, fock=fock, μ=μ, U=U,
                               include_coupling=include_coupling,
                               include_chemical_potential=include_chemical_potential,
                               include_onsite_interaction=include_onsite_interaction,
                               use_ponomarev=use_ponomarev,
                               display_progress=display_progress)
