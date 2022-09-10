import numpy as np

from synthetic_hamiltonians.operators import annihilator_for_site


def mzi_register(θ, ϕ, site, num_bosons=None, num_sites=None, use_ponomarev=True, register_index=-1):
    '''Operation corresponding to passing a time bin through the register MZI'''
    register_site = register_index % num_sites
    a1 = annihilator_for_site(register_site, num_sites, num_bosons, use_ponomarev=use_ponomarev)
    a2 = annihilator_for_site(site, num_sites, num_bosons, use_ponomarev=use_ponomarev)
    return (1j * θ / 2 * (np.exp(-1j * ϕ) * a1.dag() * a2 + np.exp(1j * ϕ) * a2.dag() * a1)).expm()


def swap_bin_into_register(site, num_bosons, num_sites, use_ponomarev=True):
    '''Swaps a photon pulse from a storage bin into the register, maintaining the phase'''
    return mzi_register(np.pi, np.pi / 2, site=site, num_bosons=num_bosons, num_sites=num_sites,
                        use_ponomarev=use_ponomarev)


def swap_bin_from_register(site, num_bosons, num_sites, use_ponomarev=True):
    '''Swaps a photon pulse from the register into a storage bin, maintaining the phase'''
    return mzi_register(np.pi, -np.pi / 2, site=site, num_bosons=num_bosons, num_sites=num_sites,
                        use_ponomarev=use_ponomarev)


def interact_register_bin(time_bin, κ, α, num_bosons, num_sites, use_ponomarev=True):
    '''Applies exp(i κ(e^-iα a1+ a2 + e^iα a2+ a1)) between the register and a time bin'''
    θ = κ * 2
    ϕ = α
    return mzi_register(θ, ϕ, time_bin, num_bosons=num_bosons, num_sites=num_sites,
                        use_ponomarev=use_ponomarev)


def interact_bin_bin(bin1, bin2, κ, α, num_bosons, num_sites, use_ponomarev=True, subdivide=False):
    '''Transfers a bin to the register, applies exp(i κ(e^-iα a1+ a2 + e^iα a2+ a1))
    between the register and the time bin, and then returns the first bin to its original space'''
    θ = κ * 2
    ϕ = α
    if subdivide:
        return [swap_bin_into_register(bin1, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev),
                mzi_register(θ, ϕ, bin2, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev),
                swap_bin_from_register(bin1, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev)]
    return swap_bin_from_register(bin1, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev) * \
           mzi_register(θ, ϕ, bin2, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev) * \
           swap_bin_into_register(bin1, num_bosons=num_bosons, num_sites=num_sites, use_ponomarev=use_ponomarev)
