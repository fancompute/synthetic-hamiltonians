from scipy.special import comb

def ponomarev_index(N, M, exact_boson_number=True):
    '''Computes 𝒩_N^M for N bosons and M sites'''
    if exact_boson_number:
        return comb(N + M - 1, N, exact=True)
    else:
        return sum(comb(n + M - 1, n, exact=True) for n in range(N, -1, -1))


def fock_to_ponomarev(sites_list, N, M, exact_boson_number=False):
    '''Converts a basis vector of the fock state (m_1 m_2 m_3 ... m_M) into Ponomarev form'''
    # Convert the occupancy to a mi >= mj array for i<j, e.g. 103020 -> 553331
    mj_list = []
    for site_index, site_occupancy in enumerate(sites_list):
        # We need to 1-index here
        for _ in range(site_occupancy):
            mj_list.append(site_index + 1)
    mj_list = sorted(mj_list, reverse=True)
    label = 1
    for i, mj in enumerate(mj_list):
        label += ponomarev_index(i + 1, M - mj, exact_boson_number=True)

    if not exact_boson_number:
        # Offset the label by the total number of labels for boson numbers greater than the true amount
        true_boson_number = sum(sites_list)
        for boson_number in range(N, true_boson_number, -1):
            label += ponomarev_index(boson_number, M, exact_boson_number=True)

    return label


def _find_largest_ponomarev_index_smaller_than(nβ, n, M):
    # find the largest 𝒩_N^m < nβ
    max_index = ponomarev_index(n, 1, exact_boson_number=True)
    # Takes care of 0 edge case
    # if max_index >= nβ:
    #     return max_index, 1
    # Loops to find the largest mn
    for m in range(1, M + 1):
        if ponomarev_index(n, m, exact_boson_number=True) < nβ:
            max_index = ponomarev_index(n, m, exact_boson_number=True)
        else:
            return max_index, m - 1
    raise RuntimeError("Index not found! (Shouldn't be here)")


def _ponomarev_to_fock_exact(index, N, M):
    mj_list = []
    nβ = index
    for n in range(N, 0, -1):
        max_index, mn = _find_largest_ponomarev_index_smaller_than(nβ, n, M)
        # print(f"{max_index=}, {mn=}, {nβ=}, {n=}")
        mj_list.append(M - mn)
        nβ = nβ - max_index
    # print(f"{mj_list=}")

    # Convert mj array back to a Fock occupancy list
    fock_list = [0] * M
    for mj in mj_list:
        fock_list[mj - 1] += 1  # convert back to 0-indexing

    return fock_list


def ponomarev_to_fock(index, N, M, exact_boson_number=False):

    assert index <= ponomarev_index(N, M, exact_boson_number=exact_boson_number), "Index outside of bounds"

    if exact_boson_number:
        return _ponomarev_to_fock_exact(index, N, M)
    else:
        # Reduce the label by decreasing boson numbers until you find one that is inside of boudns
        reduced_index = index
        for n in range(N, -1, -1):
            max_index_n_bosons = ponomarev_index(n, M, exact_boson_number=True)
            if reduced_index > max_index_n_bosons:
                reduced_index -= max_index_n_bosons
            else:
                return _ponomarev_to_fock_exact(reduced_index, n, M)
