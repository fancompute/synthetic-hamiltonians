from multiprocessing import Pool

import qutip
from scipy.linalg import logm
import qutip as qt

from tqdm import tqdm as tqdm_shell
from tqdm.notebook import tqdm as tqdm_notebook

from synthetic_hamiltonians import get_photon_occupancies


def _is_notebook():
    '''Tests to see if we are running in a jupyter notebook environment'''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


tqdm = tqdm_notebook if _is_notebook() else tqdm_shell


def _parallel_evaluate_photon_occupancies(args_tuple):
    '''Allows for evaluation of e^(i H t) * state in parallel.'''
    t, H, state, num_sites, num_bosons, use_ponomarev = args_tuple
    state_evolved = (-1j * H * t).expm() * state
    return get_photon_occupancies(state_evolved,
                                  num_sites=num_sites,
                                  num_bosons=num_bosons,
                                  use_ponomarev=use_ponomarev)


def parallel_evaluate_photon_occupancies(times, H, state,
                                         display_progress=True,
                                         pbar=None,
                                         num_workers=8,
                                         num_sites=None,
                                         num_bosons=None,
                                         use_ponomarev=True):
    '''Evaluates e^(i H t) * state in parallel.'''

    args = [(t, H, state, num_sites, num_bosons, use_ponomarev) for t in times]

    with Pool(processes=num_workers) as p:
        # results = p.starmap(_parallel_evaluate_photon_occupancies, tqdm(args, total=len(args)))
        # results = p.starmap(_parallel_evaluate_photon_occupancies, pbar(args, total=len(args)))
        results = list(tqdm(p.imap(_parallel_evaluate_photon_occupancies, args), total=len(args)))

    return results


def operator_log(op):
    '''Returns the matrix log of a qutip operator'''
    assert type(op) is qt.Qobj
    mat = op.full()
    log_mat = logm(mat)
    return qutip.Qobj(log_mat)

