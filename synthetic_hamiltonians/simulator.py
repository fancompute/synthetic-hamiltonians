import numpy as np

import qutip as qt

from synthetic_hamiltonians.mzi import interact_bin_bin
from synthetic_hamiltonians.operators import annihilator_for_site
from synthetic_hamiltonians.ponomarev import ponomarev_index
from synthetic_hamiltonians.utils import tqdm


def construct_BHH_propagator(nodes, edges, num_bosons=None,
                             μ=None, U=None,
                             include_coupling=True,
                             include_chemical_potential=False,
                             include_onsite_interaction=True,
                             normalize_μ_U_by_iteration_time=True,
                             ordering="edges",
                             use_ponomarev=True,
                             subdivide_iterations=False,
                             display_progress=False):
    '''
    Constructs a synthetic Bose-Hubbard Hamiltonian for a list of lattice sites and a list of couplings between sites
    Args:
        nodes: a list of node indices, e.g. [1,2,3,4]
        edges: a list of couplings between edges with coupling constants and phases, e.g.
               [((1,2), 0.1, np.pi), ((1,3), 0.15, np.pi/2), ...]
        num_bosons: the number of allowable bosons to support in the state space
        μ: chemical potential, normalized by iteration wall clock time by default
        U: Kerr nonlinear interaction strength, normalized by iteration wall clock time by default
        include_coupling: whether to include (κ a1.dag() a2 + H.c.) hopping terms in the Hamiltonian
        include_chemical_potential: whether to include (μ a.dag() a) terms in the Hamiltonian
        include_onsite_interaction: whether to include (U a.dag() a.dag() a a) terms in the Hamiltonian
        use_ponomarev: whether to use the reduced Ponomarev state space representation
        normalize_μ_U_by_iteration_time: whether to normalize μ and U by iteration time
        ordering: order to interact the time bins in, can be "edges", "ascending", "random" TODO
        subdivide_iterations: if True, a list of individual operations is returned in addition to the
                aggregated propagator, as well as the number of elapsed clock cycles
        display_progress: iterables will be wrapped in tqdm()
    '''
    num_time_bins_including_register = len(nodes) + 1
    if use_ponomarev:
        dim = ponomarev_index(num_bosons, num_time_bins_including_register, exact_boson_number=False) + 1
        G = qt.qeye(dim)
        if include_chemical_potential:
            H_em = qt.qzero(dim)
        if include_onsite_interaction:
            H_nl = qt.qzero(dim)
    else:
        G = qt.tensor([qt.qeye(num_bosons + 1) for _ in range(num_time_bins_including_register)])
        if include_chemical_potential:
            H_em = qt.tensor([qt.qzero(num_bosons + 1) for _ in range(num_time_bins_including_register)])
        if include_onsite_interaction:
            H_nl = qt.tensor([qt.qzero(num_bosons + 1) for _ in range(num_time_bins_including_register)])

    def interact(bin1, bin2, _κ, _α, subdivide=False):
        return interact_bin_bin(bin1, bin2, _κ, _α,
                                num_bosons=num_bosons,
                                num_sites=num_time_bins_including_register,
                                use_ponomarev=use_ponomarev,
                                subdivide=subdivide)

    clock_cycle = 0
    current_bin = 0
    operations_per_clock_cycle = []  # only used if subdivide_iterations=True
    if subdivide_iterations and (include_chemical_potential or include_onsite_interaction):
        raise NotImplementedError("Subdivide_iterations only supported in tight-binding case!")

    pbar = tqdm if display_progress else lambda x: x
    if ordering == "edges":
        iterator = edges
    else:
        raise NotImplementedError()  # TODO: add different iteration schemes

    if include_coupling:
        for ((node1, node2), κ, α) in pbar(iterator):
            previous_bin = current_bin

            # Greedily interact the nodes in ascending order of time bin
            if node1 < node2:
                G = interact(node1, node2, κ, α) * G
                if subdivide_iterations:
                    operations_per_clock_cycle.extend(interact(node1, node2, κ, α, subdivide=True))
            elif node2 < node1:
                G = interact(node1, node2, κ, -α) * G
                if subdivide_iterations:
                    operations_per_clock_cycle.extend(interact(node1, node2, κ, α, subdivide=True))
            else:
                raise ValueError(f"Can't interact node {node1} with itself!")

            # Update the current time bin position and clock cycle
            current_bin = max(node1, node2)
            if current_bin < previous_bin or min(node1, node2) < previous_bin:
                # we've had to loop back around, so update the clock cycle
                clock_cycle += 1

    if include_chemical_potential:
        for node in nodes:
            a = annihilator_for_site(site=node, num_sites=num_time_bins_including_register,
                                     num_bosons=num_bosons, use_ponomarev=use_ponomarev)
            if type(μ) is int or type(μ) is float:
                H_em += -1 * μ * a.dag() * a
            else:  # μ is a list of potentials per site
                H_em += -1 * μ[node] * a.dag() * a
        if normalize_μ_U_by_iteration_time:
            wall_time = 1
        else:
            wall_time = clock_cycle * 1

        G = (-1j * H_em * wall_time).expm() * G

    if include_onsite_interaction:
        for node in nodes:
            a = annihilator_for_site(site=node, num_sites=num_time_bins_including_register,
                                     num_bosons=num_bosons, use_ponomarev=use_ponomarev)
            H_nl += -1 * U * a.dag() * a * a.dag() * a

        if normalize_μ_U_by_iteration_time:
            wall_time = 1
        else:
            wall_time = clock_cycle * 1

        G = (-1j * H_nl * wall_time).expm() * G

    if subdivide_iterations:
        return G, operations_per_clock_cycle, clock_cycle
    else:
        return G


def construct_BHH_propagator_2d_grid(nx, ny, toroidal=False,
                                     κ=0.1,
                                     num_bosons=None,
                                     μ=None, U=None,
                                     include_coupling=True,
                                     include_chemical_potential=True,
                                     include_onsite_interaction=True,
                                     normalize_μ_U_by_iteration_time=True,
                                     ordering="edges",
                                     use_ponomarev=True,
                                     subdivide_iterations=False,
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
    return construct_BHH_propagator(nodes, edges, num_bosons=num_bosons, μ=μ, U=U,
                                    include_coupling=include_coupling,
                                    include_chemical_potential=include_chemical_potential,
                                    include_onsite_interaction=include_onsite_interaction,
                                    normalize_μ_U_by_iteration_time=normalize_μ_U_by_iteration_time,
                                    ordering=ordering,
                                    use_ponomarev=use_ponomarev,
                                    subdivide_iterations=subdivide_iterations,
                                    display_progress=display_progress)


def construct_BHH_propagator_ladder(length, width=2,
                                    hopping_phase=0.0,
                                    hopping_phase_mode="rung",  # "rung", "translation_invariant"
                                    circular=False,
                                    κ=0.1,
                                    num_bosons=None,
                                    μ=None, U=None,
                                    include_coupling=True,
                                    include_chemical_potential=True,
                                    include_onsite_interaction=True,
                                    normalize_μ_U_by_iteration_time=True,
                                    ordering="edges",
                                    use_ponomarev=True,
                                    subdivide_iterations=False,
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

    return construct_BHH_propagator(nodes, edges, num_bosons=num_bosons, μ=μ, U=U,
                                    include_coupling=include_coupling,
                                    include_chemical_potential=include_chemical_potential,
                                    include_onsite_interaction=include_onsite_interaction,
                                    normalize_μ_U_by_iteration_time=normalize_μ_U_by_iteration_time,
                                    ordering=ordering,
                                    use_ponomarev=use_ponomarev,
                                    subdivide_iterations=subdivide_iterations,
                                    display_progress=display_progress)


def construct_BHH_propagator_1d_line(num_nodes, toroidal=False,
                                     κ=0.1,
                                     α=0.0,
                                     num_bosons=None,
                                     μ=None, U=None,
                                     include_coupling=True,
                                     include_chemical_potential=True,
                                     include_onsite_interaction=True,
                                     normalize_μ_U_by_iteration_time=True,
                                     ordering="edges",
                                     use_ponomarev=True,
                                     subdivide_iterations=False,
                                     display_progress=False):
    nodes = list(range(num_nodes))
    edges = []
    for i in range(num_nodes):
        if i < num_nodes - 1 or toroidal:  # we're not at the right edge
            edges.append(((nodes[i], nodes[(i + 1) % num_nodes]), κ, α))
    return construct_BHH_propagator(nodes, edges, num_bosons=num_bosons, μ=μ, U=U,
                                    include_coupling=include_coupling,
                                    include_chemical_potential=include_chemical_potential,
                                    include_onsite_interaction=include_onsite_interaction,
                                    normalize_μ_U_by_iteration_time=normalize_μ_U_by_iteration_time,
                                    ordering=ordering,
                                    use_ponomarev=use_ponomarev,
                                    subdivide_iterations=subdivide_iterations,
                                    display_progress=display_progress)
