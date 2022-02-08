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


def ponomarev_index(N, M):
    '''Computes 𝒩_N^M for N bosons and M sites'''
    return comb(N + M - 1, N, exact=True)


def fock_to_ponomarev(sites_list, N, M):
    '''Converts a basis vector of the fock state (m_1 m_2 m_3 ... m_M) into Ponomarev form'''
    # Convert the occupancy to a mi >= mj array for i<j, e.g. 103020 -> 553331
    mj_list = []
    for site_index, site_occupancy in enumerate(sites_list):
        # We need to 1-index here
        for _ in range(site_occupancy):
            mj_list.append(site_index + 1)
    mj_list = sorted(mj_list, reverse=True)
    # print(f"{mj_list=}")
    label = 1
    for i, mj in enumerate(mj_list):
        # print(i+1, M-mj, ponomarev_index(i + 1, M - mj))
        # print(f"mj={i+1}, {M-mj=}, index={ponomarev_index(i + 1, M - mj)}")
        label += ponomarev_index(i + 1, M - mj)
    return label


def ponomarev_to_fock(index, N, M):
    # find the largest 𝒩_N^m < nβ
    def find_largest_ponomarev_index_smaller_than(nβ, n):
        max_index = ponomarev_index(n, 1)
        # Takes care of 0 edge case
        # if max_index >= nβ:
        #     return max_index, 1
        # Loops to find the largest mn
        for m in range(1, M + 1):
            if ponomarev_index(n, m) < nβ:
                max_index = ponomarev_index(n, m)
            else:
                return max_index, m - 1
        raise RuntimeError("Index not found! (Shouldn't be here)")
        # i = 1
        # max_index = ponomarev_index(n, 1)
        # while max_index < nβ:
        #     i += 1
        #     max_index = ponomarev_index(n, i)
        # return max_index, i

    mj_list = []
    nβ = index
    for n in range(N, 0, -1):
        max_index, mn = find_largest_ponomarev_index_smaller_than(nβ, n)
        # print(f"{max_index=}, {mn=}, {nβ=}, {n=}")
        mj_list.append(M - mn)
        nβ = nβ - max_index
    # print(f"{mj_list=}")

    # Convert mj array back to a Fock occupancy list
    fock_list = [0] * M
    for mj in mj_list:
        fock_list[mj - 1] += 1  # convert back to 0-indexing

    return fock_list
