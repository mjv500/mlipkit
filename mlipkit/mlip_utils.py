import numpy as np
from pathlib import Path

from ase.atoms import Atoms

from .utils import mae, rmse, R2, flatten, decapitalize


def extract_prop_from_single_ase(structure):
    '''Function to extract energy (per atom), forces and stress from an ase trajectory

    Parameters
    ----------
    structures: ase.atoms.Atoms or list of ase.atoms.Atoms
       trajectory

    Returns
    -------
    energy: float
            (nuclear potential) energy PER ATOM (in eV/atom)
    forces: 2D np.array of floats
            forces with shape natoms x 3 (in eV/Angst)
    stress: 2D np.array of floats
            stress tensor (in eV/Angst^3)

    Notes
    -----
    The convention for the stress is that the stress tensor element are:
    sigma = + dE/dn (n=strain) (note the sign!)

    '''
    energy = structure.get_potential_energy()/len(structure)
    forces = structure.get_forces()
    stress = structure.get_stress()

    return energy, forces, stress

def extract_prop_from_ase(structures):
    '''Function to extract energy (per atom), forces and stress from an ase trajectory
    
    Parameters
    ----------
    structures: ase.atoms.Atoms or list of ase.atoms.Atoms
       trajectory 
        
    Returns
    -------
    energy: list fof floats
            (nuclear potential) energy PER ATOM of each configuration (in eV/atom)
    forces: list of 2D np.arrays of floats
            forces with shape nconfs x natoms x 3 (in eV/Angst)
    stress: list of 2D np.arrays of floats
            stress tensor for each configuration (in eV/Angst^2)
        
    Notes
    -----
    The convention for the stress is that the stress tensor element are:
    sigma = + dE/dn (n=strain) (note the sign!)
    
    '''
    if isinstance(structures, Atoms):
        return extract_prop_from_single_ase(structures)
        
    else:
        res = [extract_prop_from_single_ase(x) for x in structures]
        energy = [x[0] for x in res]
        forces = [x[1] for x in res]
        stress = [x[2] for x in res]
        return energy, forces, stress

def make_comparison(structures1, 
                    structures2, 
                    props='all', 
                    make_file=False, 
                    dir='./',
                    outfile_pref='', 
                    units=None):
    '''Create the comparison files for energy, forces and stress starting from lists of ase Atoms objects.
    
    Parameters
    ----------
    structures1: ase.atoms.Atoms or list of ase.atoms.Atoms
        mandatory when is_ase1 = True (ignored otherwise); (list of) ase
        Atoms object(s) with the true values
    structures2: ase.atoms.Atoms or list of ase.atoms.Atoms
        mandatory when is_ase2 = True (ignored otherwise); (list of) ase
        Atoms object(s) with the ML values
    props: str or list of {'energy', 'forces', 'stress', 'all'}
        if a list is given containing 'all', all three properties will be
        considered, independent on the other elements of the list
    make_file: bool
        - True: create a comparison file
    dir: str
        directory the output file will be saved (if make_file=True)
    outfile_pref: str
        the output file will be named [outfile_pref][Property]_comparison.dat 
        (if make_file=True) e.g.: with outfile_pref = 'MLIP-', for the energy 
        the name would be: MLIP-Energy_comparison.dat 
    units: dict, default: {'energy': 'eV/at', 'forces':'eV/Angs', 'stress':'eV/Angst^3'}
        dictionary with key-value pairs like prop-unit with prop in 
        ['energy', 'forces', 'stress'] and value being a string with the unit
        to print for the respective property. If None, the respective units
        will be eV/at, eV/Angs and eV/Angst^3. This only affects the header of the output file.

    Returns
    -------
    errs: dict
    dictionary whose key/vlaue elements are property/errors, where property can be 
    {'energy', 'forces', 'stress'}, and errors is a list [rmse, mae, R2]. 
        
    '''
    
    if make_file == True:
        dir = Path(dir)
                    
    if isinstance(props, str):
        props = [props]
        
    if not all([x in ['all', 'energy', 'forces', 'stress'] for x in props]):
        raise ValueError("Please give a value or a list of values chosen from ['energy', 'forces', 'stress', 'all']")
    
    if not isinstance(props, list):
        props = [props]
    
    if props == ['all'] or (isinstance(props, list) and 'all' in props):
        props =  ['energy', 'forces', 'stress']
    
    if units == None:
        units = dict(energy='eV/at', forces='eV/$\mathrm{\AA}$', stress='eV/$\mathrm{\AA}^3$')
        
    prop_numbs = dict(energy = 0, forces = 1, stress = 2)
    
    if not len(structures1) == len(structures2):
        raise ValueError(f"You gave a different number of true and ML structures!")
    
    # Retrieve the data
    ext1 = [flatten(x) for x in extract_prop_from_ase(structures1)]
    ext2 = [flatten(x) for x in extract_prop_from_ase(structures2)]

    dir = Path(dir)
    # Compute errors and write data on files
    errs = dict()
    for prop in props:
        i = prop_numbs[prop]
        filename = dir.joinpath(f'{outfile_pref}{prop.capitalize()}_comparison.dat')
        mae2 = mae(ext1[i], ext2[i])
        rmse2 = rmse(ext1[i], ext2[i])
        R22 = R2(ext1[i], ext2[i])
        errs[prop] = [rmse2, mae2, R22]
        
        if make_file == True:
            print(f'printing in {filename.absolute()}')
            text = f'# rmse: {rmse2:.5f} {units[prop]},    mae: {mae2:.5f} {units[prop]}    R2: {R22:.5f}\n'
            text += f'#  True {decapitalize(prop)}           Predicted {decapitalize(prop)}\n'
            for x, y in zip(ext1[i], ext2[i]):
                text += f'{x:.20f}  {y:.20f}\n'
            with open(filename.absolute(), 'w') as fl:
                fl.write(text)
    return errs
