import numpy as np

import qutip as qt

from synthetic_hamiltonians.operators import annihilator_for_site
from synthetic_hamiltonians.ponomarev import ponomarev_index
from synthetic_hamiltonians.utils import tqdm


def make_BH_Hamiltonian(nodes, edges, num_bosons=None, μ=None, U=None,
                        include_coupling=True,
                        include_chemical_potential=False,
                        include_onsite_interaction=True,
                        add_extraneous_register_node=False,
                        use_ponomarev=True,
                        display_progress=False):
    '''
    Returns the actual Bose-Hubbard Hamiltonian for a list of lattice sites and a list of couplings between sites
    Args:
        nodes: a list of node indices, e.g. [1,2,3,4]
        edges: a list of couplings between edges with coupling constants and phases, e.g.
            [((1,2), 0.1, np.pi), ((1,3), 0.15, np.pi/2), ...]
        num_bosons: the number of allowable bosons to support in the state space
        μ: chemical potential
        U: onsite interaction strength
        include_coupling: whether to include (κ a1.dag() a2 + H.c.) hopping terms in the Hamiltonian
        include_chemical_potential: whether to include (μ a.dag() a) terms in the Hamiltonian
        include_onsite_interaction: whether to include (U a.dag() a.dag() a a) terms in the Hamiltonian
        add_extraneous_register_node: whether to include a non-interacting extra site at index=-1 (for the purposes
            of comparing against the synthetically constructed version)
        use_ponomarev: whether to use the reduced Ponomarev state space representation
        display_progress: iterables will be wrapped in tqdm()
    '''

    if add_extraneous_register_node:
        # Add an additional site to represent the register time bin
        nodes.append(max(nodes) + 1)

    if use_ponomarev:
        dim = ponomarev_index(num_bosons, len(nodes), exact_boson_number=False) + 1
        H = qt.qzero(dim)
    else:
        H = qt.tensor([qt.qzero(num_bosons + 1) for _ in nodes])

    if include_coupling:
        iterator = tqdm if display_progress else lambda x: x
        for ((node1, node2), κ, α) in iterator(edges):
            a1 = annihilator_for_site(site=node1, num_sites=len(nodes), num_bosons=num_bosons,
                                      use_ponomarev=use_ponomarev)
            a2 = annihilator_for_site(site=node2, num_sites=len(nodes), num_bosons=num_bosons,
                                      use_ponomarev=use_ponomarev)
            # a1 = get_annihilator_index(index=node1, num_operators=len(nodes), fock=fock)
            # a2 = get_annihilator_index(index=node2, num_operators=len(nodes), fock=fock)
            H += -1 * κ * (np.exp(-1j * α) * a1.dag() * a2 + np.exp(1j * α) * a2.dag() * a1)

    if include_chemical_potential:
        for node in nodes:
            a = annihilator_for_site(site=node, num_sites=len(nodes), num_bosons=num_bosons,
                                     use_ponomarev=use_ponomarev)
            # a = get_annihilator_index(index=node, num_operators=len(nodes), fock=fock)
            if type(μ) is int or type(μ) is float:
                H += -1 * μ * a.dag() * a
            else:  # μ is a list of potentials per site
                H += -1 * μ[node] * a.dag() * a

    if include_onsite_interaction:
        for node in nodes:
            a = annihilator_for_site(site=node, num_sites=len(nodes), num_bosons=num_bosons,
                                     use_ponomarev=use_ponomarev)
            # a = get_annihilator_index(index=node, num_operators=len(nodes), fock=fock)
            H += -1 * U * a.dag() * a * a.dag() * a
            # H += -1 * U * a.dag() * a.dag() * a * a

    return H


def BHH_2d_grid(nx, ny, toroidal=False,
                κ=0.1,
                num_bosons=None,
                μ=None, U=None,
                include_coupling=True,
                include_chemical_potential=True,
                include_onsite_interaction=True,
                add_extraneous_register_node=False,
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
            elif toroidal == True or toroidal == "x":
                edges.append(((nodes_xy[y, x], nodes_xy[y, (x + 1) % nx]), κ, 0))
            if y < ny - 1:  # we're not at the bottom edge
                edges.append(((nodes_xy[y, x], nodes_xy[y + 1, x]), κ, 0))
            elif toroidal == True or toroidal == "y":
                edges.append(((nodes_xy[y, x], nodes_xy[(y + 1) % ny, x]), κ, 0))
    return make_BH_Hamiltonian(nodes, edges, num_bosons=num_bosons, μ=μ, U=U,
                               include_coupling=include_coupling,
                               include_chemical_potential=include_chemical_potential,
                               include_onsite_interaction=include_onsite_interaction,
                               add_extraneous_register_node=add_extraneous_register_node,
                               use_ponomarev=use_ponomarev,
                               display_progress=display_progress)


def BHH_ladder(length, width=2,
               hopping_phase=0.0,
               hopping_phase_mode="rung",  # "rung", "translation_invariant"
               circular=False,
               κ=0.1,
               num_bosons=None,
               μ=None, U=None,
               include_coupling=True,
               include_chemical_potential=True,
               include_onsite_interaction=True,
               add_extraneous_register_node=False,
               use_ponomarev=True,
               display_progress=False):
    if hopping_phase_mode not in ["rung", "translation_invariant"]:
        raise ValueError("Bad value for argument hopping_phase_mode")
    num_nodes = length * width
    nodes = list(range(num_nodes))
    nodes_xy = np.array(nodes).reshape((length, width))
    edges = []
    for y in range(length):
        for x in range(width):
            if x < width - 1:  # we're not at the right edge
                if hopping_phase_mode == "rung":
                    α = y * hopping_phase
                else:
                    α = 0
                edges.append(((nodes_xy[y, x], nodes_xy[y, x + 1]), κ, α))
            # elif toroidal:
            #     edges.append(((nodes_xy[y, x], nodes_xy[y, (x + 1) % nx]), κ, 0))
            if y < length - 1 or circular:
                if hopping_phase_mode == "translation_invariant":
                    if x == 0:  # left ladder leg
                        α = -1 * hopping_phase / 2
                    else:  # right ladder leg
                        α = hopping_phase / 2
                else:
                    α = 0
                edges.append(((nodes_xy[y, x], nodes_xy[(y + 1) % length, x]), κ, α))

    return make_BH_Hamiltonian(nodes, edges, num_bosons=num_bosons, μ=μ, U=U,
                               include_coupling=include_coupling,
                               include_chemical_potential=include_chemical_potential,
                               include_onsite_interaction=include_onsite_interaction,
                               add_extraneous_register_node=add_extraneous_register_node,
                               use_ponomarev=use_ponomarev,
                               display_progress=display_progress)


def BHH_1d_line(num_nodes, toroidal=False,
                κ=0.1,
                α=0.0,
                num_bosons=None,
                μ=None, U=None,
                include_coupling=True,
                include_chemical_potential=True,
                include_onsite_interaction=True,
                add_extraneous_register_node=False,
                use_ponomarev=True,
                display_progress=False):
    nodes = list(range(num_nodes))
    edges = []
    for i in range(num_nodes):
        if i < num_nodes - 1 or toroidal:  # we're not at the right edge
            edges.append(((nodes[i], nodes[(i + 1) % num_nodes]), κ, α))
    return make_BH_Hamiltonian(nodes, edges, num_bosons=num_bosons, μ=μ, U=U,
                               include_coupling=include_coupling,
                               include_chemical_potential=include_chemical_potential,
                               include_onsite_interaction=include_onsite_interaction,
                               add_extraneous_register_node=add_extraneous_register_node,
                               use_ponomarev=use_ponomarev,
                               display_progress=display_progress)
