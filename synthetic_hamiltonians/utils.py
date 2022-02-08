from tqdm import tqdm as tqdm_shell
from tqdm.notebook import tqdm as tqdm_notebook


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
