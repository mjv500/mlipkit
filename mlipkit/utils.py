import numpy as np
import numbers
from pathlib import Path

from scipy.stats import linregress
import hashlib
from ase.atoms import Atoms



def decapitalize(string):
    """
    Converts only the first character of a string to lowercase.

    Args:
        string (str): The input string.

    Returns:
        str: The string with the first character in lowercase.
    """
    string = string[0].lower() + string[1:]
    return string


def mae(data1, data2):
    """
    Computes the Mean Absolute Error (MAE) between two datasets.

    Args:
        data1 (list, np.ndarray, or number): Reference (true) values.
        data2 (list, np.ndarray, or number): Observed (predicted) values.

    Returns:
        float: The calculated MAE.

    Raises:
        TypeError: If inputs are not numbers, lists, or numpy arrays.
        ValueError: If the two datasets have different lengths.
    """

    if isinstance(data1, numbers.Number):
        data1 = np.array([data1])
    if not isinstance(data1, list) and not isinstance(data1, type(np.array([42]))):
        raise TypeError('data1 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data1]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data1, list):
        data1 = np.array(data1)
    
    if isinstance(data2, numbers.Number):
        data2 = np.array([data2])
    if not isinstance(data2, list) and not isinstance(data2, type(np.array([42]))):
        raise TypeError('data2 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data2]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data2, list):
        data2 = np.array(data2)        
    
    if len(data1) != len(data2):
        raise ValueError('The two array must have the same size!')
    
    ae = np.absolute(data1 - data2)
    mae = np.mean(ae)
    #print(f'data1: {data1}, data2: {data2}, ae: {ae}, mae: {mae}')
    return mae


def rmse(data1, data2):
    """
    Computes the Root Mean Square Error (RMSE) between two datasets.

    Args:
        data1 (list, np.ndarray, or number): Reference (true) values.
        data2 (list, np.ndarray, or number): Observed (predicted) values.

    Returns:
        float: The calculated RMSE.

    Raises:
        TypeError: If inputs are not numbers, lists, or numpy arrays.
        ValueError: If the two datasets have different lengths.
    """
    
    if isinstance(data1, numbers.Number):
        data1 = np.array([data1])
    if not isinstance(data1, list) and not isinstance(data1, type(np.array([42]))):
        raise TypeError('data1 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data1]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data1, list):
        data1 = np.array(data1)
    
    if isinstance(data2, numbers.Number):
        data2 = np.array([data2])
    if not isinstance(data2, list) and not isinstance(data2, type(np.array([42]))):
        raise TypeError('data2 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data2]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data2, list):
        data2 = np.array(data2)        
    
    if len(data1) != len(data2):
        raise ValueError('The two array must have the same size!')
    
    se = (data1 - data2) ** 2
    mse = np.mean(se)
    rmse = mse ** 0.5
    #print(f'data1: {data1}, data2: {data2}, se: {se}, mse: {mse}, rmse: {rmse}')
    return rmse

def R2(data1, data2):
    """
    Computes the Coefficient of Determination (R²) using linear regression.

    Args:
        data1 (list or np.ndarray): Reference (true) data.
        data2 (list or np.ndarray): Observed (predicted) data.

    Returns:
        float: The R² value.
    """
    
    _, _, r_value, _, _ = linregress(data1, data2)
    return r_value**2

def flatten(l):
    """
    Recursively flattens a nested list or numpy array into a single list.

    Args:
        l (list or np.ndarray): The nested structure to flatten.

    Returns:
        list: A 1D list containing all individual elements.

    Raises:
        TypeError: If the input is not a list or numpy array.
    """
    
    if not isinstance(l, (list, np.ndarray)):
        raise TypeError("The object given is not a list!")
    
    res = []
    for x in l:
        if isinstance(x, (list, np.ndarray)):
            res.extend(flatten(x))  # Always recursively flatten lists
        else:
            res.append(x)  # Append non-list elements directly
    
    return res

def hash_multiple_files(filepaths):
    """
    Generates a unique SHA256 hash based on the content and names of multiple files.

    This is used to "lock" a model to its specific potential weights. The file 
    order is preserved to ensure deterministic hashing.

    Args:
        filepaths (list of str or Path): The files to be hashed.

    Returns:
        str: The hex digest of the SHA256 hash.
    """
    
    sha = hashlib.sha256()
    
    # Ensure consistent ordering
    for path in filepaths:
        path = Path(path)
        sha.update(path.name.encode())  # Include file name to distinguish identical content across files
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)

    return sha.hexdigest()

def atoms_hash(structures: Atoms, include_results=True, precision=1e-6) -> str:
    """
    Generates a SHA256 hash for a sequence of ASE Atoms objects.

    The hash considers atomic numbers, cell parameters, and positions. 
    Optionally includes calculator results (energy, forces, stress) to 
    distinguish between identical geometries with different metadata.

    Args:
        structures (ase.Atoms or list of ase.Atoms): The configurations to hash.
        include_results (bool): Whether to include energy/forces/stress in the hash.
        precision (float): Numerical precision for rounding before hashing to 
            avoid noise-induced discrepancies.

    Returns:
        str: The hex digest of the SHA256 hash.
    """
    
    if isinstance(structures, Atoms):
        structures = [structures]

    def arr_to_bytes(arr):
        arr = np.array(arr).round(int(-np.log10(precision)))
        return arr.tobytes()
    
    h = hashlib.sha256()
    
    for structure in structures: 
        # Atomic numbers
        h.update(arr_to_bytes(structure.numbers))
        
        # Cell
        h.update(arr_to_bytes(structure.cell.array))
        
        # Positions
        h.update(arr_to_bytes(structure.positions))
        
        if include_results and structure.calc is not None:
            try:
                e = structure.get_potential_energy()
                h.update(str(round(float(e), 6)).encode())
            except Exception:
                pass
            try:
                f = structure.get_forces()
                h.update(arr_to_bytes(f))
            except Exception:
                pass
            try:
                s = structure.get_stress()
                h.update(arr_to_bytes(s))
            except Exception:
                pass
    
    return h.hexdigest()


def find_min_dist(trajectory):
    """
    Identifies the minimum interatomic distance found across a trajectory.

    This function calculates all pairwise distances for every frame, accounting 
    for periodic boundary conditions (Minimum Image Convention). It ignores 
    self-distances (diagonal elements) to find the smallest physical 
    separation between distinct atoms.

    Args:
        trajectory (list of ase.Atoms): A sequence of ASE configurations.

    Returns:
        float: The absolute minimum distance (Angstrom) found.
    """
    if isinstance(trajectory, Atoms):
        trajectory = [trajectory]
        
    global_min = np.inf
    
    for atoms in trajectory:
        # Get the full distance matrix (N x N)
        dists = atoms.get_all_distances(mic=True)
        
        # Replace 0.0 on the diagonal with infinity so .min() ignores it
        np.fill_diagonal(dists, np.inf)
        
        frame_min = dists.min()
        if frame_min < global_min:
            global_min = frame_min
            
    return float(global_min)


def space(n):
    return ' '*n