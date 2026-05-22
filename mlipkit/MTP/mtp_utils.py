import numpy as np
from pathlib import Path
from copy import deepcopy as cp
from subprocess import run
import shutil

from ase.io import read, write
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammpsrun import LAMMPS

from ..utils import mae, rmse, R2, decapitalize, flatten
from ..mlip_utils import extract_prop_from_ase, extract_prop_from_single_ase


def train_pot_from_ase(mlip_bin,
                       untrained_pot_file_dir,
                       mtp_level,
                       min_dist,
                       max_dist,
                       radial_basis_size,
                       radial_basis_type,
                       train_set,
                       dir,
                       params,
                       mpirun='',
                       final_evaluation=True):
    """
    Trains an MTP model directly from an ASE trajectory.

    Args:
        mlip_bin (str): Path to the MLIP binary.
        untrained_pot_file_dir (str): Path to the directory containing the untrained MTP init files (.mtp).
        mtp_level (int): Level of the MTP model to train (e.g., 2, 4, 6... 28).
        min_dist (float or str): Minimum distance between atoms in the system in Angstroms, or 'find' to calculate it automatically.
        max_dist (float): Cutoff radius for the radial part in Angstroms.
        radial_basis_size (int): Number of basis functions to use for the radial part.
        radial_basis_type (str): Type of basis functions to use for the radial part.
        train_set (list of ase.Atoms): List of ASE Atoms objects. Energy, forces, and stresses must be stored in each object.
        dir (str): Path to the directory where the training will run and output will be saved.
        params (dict): Dictionary containing the training flags. Accepted keys include:
            - ene_weight (float, optional): Weight of energies in the fitting. Defaults to 1.
            - for_weight (float, optional): Weight of forces in the fitting. Defaults to 0.01.
            - str_weight (float, optional): Weight of stresses in the fitting. Defaults to 0.001.
            - sc_b_for (float, optional): Weight multiplier for configurations near equilibrium. Defaults to 0.
            - val_cfg (str, optional): Filename with configurations to validate.
            - max_iter (int, optional): Maximal number of iterations. Defaults to 1000.
            - cur_pot_n (str, optional): If provided, saves potential on each iteration with this name.
            - trained_pot_name (str, optional): Filename for the trained potential. Defaults to "Trained.mtp".
            - bfgs_tol (float, optional): Stops if error drops by a factor smaller than this over 50 BFGS iterations. Defaults to 1e-3.
            - weighting (str, optional): Config weighting strategy ('vibrations', 'molecules', 'structures'). Defaults to 'vibrations'.
            - init_par (str, optional): Parameter initialization strategy ('random', 'same'). Defaults to 'random'.
            - skip_preinit (bool, optional): Skips the 75 iterations done when parameters are not given.
            - up_mindist (bool, optional): Updates the mindist parameter with actual minimal interatomic distance in the training set.
        mpirun (str, optional): Command for MPI parallelization (e.g., 'mpirun'). Defaults to ''.
        final_evaluation (bool, optional): If True, evaluates the training set with the trained potential. Defaults to True.

    Returns:
        tuple or str: If final_evaluation is True, returns a tuple containing:
            - trained_pot_file_path (Path): Path to the trained .mtp file.
            - ml_trainset (list of ase.Atoms): ASE trajectory evaluated with the MLIP.
            - errs (dict): RMSE, MAE, and R2 errors for energy, forces, and stress.
            If False, returns only the trained_pot_file_path (Path).
    """
    dir = Path(dir)
    mlip_bin = Path(mlip_bin)
    untrained_pot_file_dir = Path(untrained_pot_file_dir)    
    cfg_path = dir.joinpath('TrainSet.cfg')
    conv_ase_to_mlip2(atoms=train_set,
                      out_path=cfg_path.absolute(),
                      props=True)
    [at.get_chemical_symbols() for at in train_set]
    species_count = len(set(flatten([at.get_chemical_symbols() for at in train_set])))
    if min_dist == 'find':
            min_dist = find_min_dist(train_set)
    results  = train_pot(mlip_bin=mlip_bin.absolute(),
                         untrained_pot_file_dir=untrained_pot_file_dir.absolute(),
                         mtp_level=mtp_level,
                         species_count=species_count,
                         min_dist=min_dist,
                         max_dist=max_dist,
                         radial_basis_size=radial_basis_size,
                         radial_basis_type=radial_basis_type,
                         train_set_path=cfg_path.absolute(), 
                         dir=dir.absolute(),
                         params=params,
                         mpirun=mpirun,
                         final_evaluation=final_evaluation)
    if final_evaluation == False:
        trained_pot_file_path = results
        return trained_pot_file_path
    else:
        trained_pot_file_path = results[0]
        ml_trainsetpath = results[1]
        errs = results[2]
        elemlist = [x for x in set(flatten([conf.get_chemical_symbols() for conf in train_set]))]
        ml_trainset = conv_mlip2_to_ase(ml_trainsetpath, props=True, elemlist=elemlist)
        return trained_pot_file_path, ml_trainset, errs
    
def conv_ase_to_mlip2(atoms, out_path, props=True):
    """
    Converts a trajectory of ASE Atoms objects into a .cfg file for MLIP-2.

    Args:
        atoms (list of ase.Atoms): List of the ASE configurations.
        out_path (str or Path): Path to the output file (must include the .cfg extension).
        props (bool, optional): If True, copies energy, stress, and forces (must be computed beforehand). Defaults to True.
    """
    text = conv_ase_to_mlip2_text(atoms=atoms, props=props)
    with open(out_path.absolute(), 'w') as fl:
        fl.write(text)

def conv_single_ase_to_mlip2(atoms, out_path, props=True):
    """
    Converts a single ASE Atoms object into a .cfg file for MLIP-2.

    Args:
        atoms (ase.Atoms): The ASE configuration.
        out_path (str or Path): Path to the output file (must include the .cfg extension).
        props (bool, optional): If True, copies energy, stress, and forces (must be computed beforehand). Defaults to True.
    """
    text = conv_single_ase_to_mlip2_text(atoms=atoms, props=props)

    with open(out_path.absolute(), 'w') as fl:
        fl.write(text)

def conv_single_ase_to_mlip2_text(atoms, props=True):
    """
    Converts a single ASE Atoms object into a formatted text string for a .cfg file.

    Args:
        atoms (ase.Atoms): The ASE configuration.
        props (bool, optional): If True, copies energy, stress, and forces. Defaults to True.

    Returns:
        str: The formatted text string representing the MLIP .cfg configuration.
    """

    text = ''
    natom = len(atoms)
    cell = atoms.get_cell()
    if props == True:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress() * atoms.get_volume() # MTP uses stress multiplied by the volume
    at_syms = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    elemlist = list(set(at_syms))
    for i in range(len(elemlist)):
        elemlist[i] = elemlist[i].capitalize()
    elemlist.sort()
    nelem = len(elemlist)
    elems = dict()
    for i, el in enumerate(elemlist):
        elems[f'{el}'] = i

    # WRITE TEXT
    text += f'BEGIN_CFG\n'
    text += f'Size\n'
    text += f'{natom}\n'
    text += f'Supercell\n'
    text += f'{cell[0][0]}\t{cell[0][1]}\t{cell[0][2]}\n'
    text += f'{cell[1][0]}\t{cell[1][1]}\t{cell[1][2]}\n'
    text += f'{cell[2][0]}\t{cell[2][1]}\t{cell[2][2]}\n'
    text += f'AtomData:\tid\ttype\tcartes_x\tcartes_y\tcartes_z\tfx\tfy\tfz\n'
    for i, atm in enumerate(at_syms):
        text += f'{i+1}\t'
        text += str(elems[atm]) + '\t'
        #text += f'{elems[f'{atm}']}\t'
        text += f'{positions[i][0]:15.20f}\t'
        text += f'{positions[i][1]:15.20f}\t'
        text += f'{positions[i][2]:15.20f}\t'
        if props == True:
            text += f'{forces[i][0]:15.20f}\t'
            text += f'{forces[i][1]:15.20f}\t'
            text += f'{forces[i][2]:15.20f}\t'
        else:
            text += f'{0:15.20f}\t' # fake forces
            text += f'{0:15.20f}\t'
            text += f'{0:15.20f}\t'
        text += f'\n'
    if props == True:
        text += f'Energy\n'
        text += f'{energy:15.20f}\n'
        text += f'PlusStress:\txx\tyy\tzz\tyz\txz\txy\n'
        text += f'{-stress[0]:15.20f}\t{-stress[1]:15.20f}\t{-stress[2]:15.20f}\t{-stress[3]:15.20f}\t{-stress[4]:15.20f}'\
              + f'\t{-stress[5]:15.20f}\n'
    text += f'END_CFG\n\n'
    return text


def conv_ase_to_mlip2_text(atoms, props=True):
    """
    Converts a trajectory of ASE Atoms objects into a formatted text string for a .cfg file.

    Args:
        atoms (ase.Atoms or list of ase.Atoms): The ASE configuration(s).
        props (bool, optional): If True, copies energy, stress, and forces. Defaults to True.

    Returns:
        str: The formatted text string representing the MLIP .cfg configurations.
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]
        
    text = ''
    for x in atoms:
        conf = x
        natom = len(conf)
        cell = conf.get_cell()
        if props == True:
            energy = conf.get_potential_energy()
            forces = conf.get_forces()
            stress = conf.get_stress() * conf.get_volume() # MTP uses stress multiplied by the volume
        at_syms = conf.get_chemical_symbols()
        positions = conf.get_positions()
        elemlist = list(set(at_syms))
        for i in range(len(elemlist)):
            elemlist[i] = elemlist[i].capitalize()
        elemlist.sort()
        nelem = len(elemlist)
        elems = dict()
        for i, el in enumerate(elemlist):
            elems[f'{el}'] = i
        # WRITE TEXT
        text += f'BEGIN_CFG\n'
        text += f'Size\n'
        text += f'{natom}\n'
        text += f'Supercell\n'
        text += f'{cell[0][0]}\t{cell[0][1]}\t{cell[0][2]}\n'
        text += f'{cell[1][0]}\t{cell[1][1]}\t{cell[1][2]}\n'
        text += f'{cell[2][0]}\t{cell[2][1]}\t{cell[2][2]}\n'
        text += f'AtomData:\tid\ttype\tcartes_x\tcartes_y\tcartes_z\tfx\tfy\tfz\n'
        for i, atm in enumerate(at_syms):
            text += f'{i+1}\t'
            text += str(elems[atm]) + '\t'
            #text += f'{elems[f'{atm}']}\t'
            text += f'{positions[i][0]:15.20f}\t'
            text += f'{positions[i][1]:15.20f}\t'
            text += f'{positions[i][2]:15.20f}\t'
            if props == True:
                text += f'{forces[i][0]:15.20f}\t'
                text += f'{forces[i][1]:15.20f}\t'
                text += f'{forces[i][2]:15.20f}\t'
            else:
                text += f'{0:15.20f}\t' # fake forces
                text += f'{0:15.20f}\t'
                text += f'{0:15.20f}\t'
            text += f'\n'
        if props == True:
            text += f'Energy\n'
            text += f'{energy:15.20f}\n'
            text += f'PlusStress:\txx\tyy\tzz\tyz\txz\txy\n'
            text += f'{-stress[0]:15.20f}\t{-stress[1]:15.20f}\t{-stress[2]:15.20f}\t{-stress[3]:15.20f}\t{-stress[4]:15.20f}'\
                  + f'\t{-stress[5]:15.20f}\n'
        text += f'END_CFG\n\n'
    return text

def train_pot(mlip_bin, 
              untrained_pot_file_dir,
              mtp_level,
              min_dist,
              max_dist,
              species_count,
              radial_basis_size,
              radial_basis_type,
              train_set_path,
              dir,
              params,
              mpirun='',
              final_evaluation=False):
    """
    Core function to execute the MLIP-2 training process via command line.

    Args:
        mlip_bin (str): Path to the MLIP binary.
        untrained_pot_file_dir (str): Path to the directory containing the untrained MTP init files (.mtp).
        mtp_level (int): Level of the MTP model to train.
        min_dist (float): Minimum distance between atoms in the system in Angstroms.
        max_dist (float): Cutoff radius for the radial part in Angstroms.
        species_count (int): Number of distinct chemical elements in the dataset.
        radial_basis_size (int): Number of basis functions to use for the radial part.
        radial_basis_type (str): Type of basis functions to use for the radial part.
        train_set_path (str): Path to the training set .cfg file.
        dir (str): Path to the directory where the training will run.
        params (dict): Dictionary containing the training flags. Accepted keys include:
            - ene_weight (float, optional): Weight of energies in the fitting. Defaults to 1.
            - for_weight (float, optional): Weight of forces in the fitting. Defaults to 0.01.
            - str_weight (float, optional): Weight of stresses in the fitting. Defaults to 0.001.
            - sc_b_for (float, optional): Weight multiplier for configurations near equilibrium. Defaults to 0.
            - val_cfg (str, optional): Filename with configurations to validate.
            - max_iter (int, optional): Maximal number of iterations. Defaults to 1000.
            - cur_pot_n (str, optional): If provided, saves potential on each iteration with this name.
            - trained_pot_name (str, optional): Filename for the trained potential. Defaults to "Trained.mtp".
            - bfgs_tol (float, optional): Stops if error drops by a factor smaller than this over 50 BFGS iterations. Defaults to 1e-3.
            - weighting (str, optional): Config weighting strategy ('vibrations', 'molecules', 'structures'). Defaults to 'vibrations'.
            - init_par (str, optional): Parameter initialization strategy ('random', 'same'). Defaults to 'random'.
            - skip_preinit (bool, optional): Skips the 75 iterations done when parameters are not given.
            - up_mindist (bool, optional): Updates the mindist parameter with actual minimal interatomic distance in the training set.
        mpirun (str, optional): Command for MPI parallelization. Defaults to ''.
        final_evaluation (bool, optional): If True, runs an evaluation on the dataset post-training. Defaults to False.

    Returns:
        tuple or Path: If final_evaluation is True, returns (trained_pot_file_path, out_path, errs).
        Otherwise, returns just trained_pot_file_path.
    """
    
    
    def get_flags(params):
        flags = dict(ene_weight = '--energy-weight',
                     for_weight = '--force-weight',
                     str_weight = '--stress-weight',
                     sc_b_for = '--scale-by-force',
                     val_cfg = '--valid_cfgs',
                     max_iter = '--max-iter',
                     cur_pot_n = '--curr-pot-name',
                     trained_pot_name = '--trained-pot-name',
                     bfgs_tol = '--bfgs-conv-tol',
                     weighting = '--weighting',
                     init_par = '--init-params',
                     skip_preinit = '--skip-preinit',
                     up_mindist = '--update-mindist')

        cmd = ''
        for par in list(params.keys()):
            if par == 'skip_preinit':
                if params[par] == True:
                    cmd = f'{cmd} {flags[par]}'
                continue
            elif par == 'up_mindist':
                if params[par] == True:
                    cmd = f'{cmd} {flags[par]}'
                continue
            elif par in list(flags.keys()):
                cmd = f'{cmd} {flags[par]}={params[par]}'         
        return cmd
    
    dir = Path(dir)
    mlip_bin = Path(mlip_bin)
    untrained_pot_file_dir = Path(untrained_pot_file_dir)
    train_set_path = Path(train_set_path)
                 
    if 'trained_pot_name' not in list(params.keys()):
        params['trained_pot_name'] = 'pot.mtp'
    flags = get_flags(params)
    make_mtp_file(sp_count=species_count,
                  mind=min_dist,
                  maxd=max_dist,
                  rad_bas_sz=radial_basis_size,
                  rad_bas_type=radial_basis_type, 
                  lev=mtp_level, 
                  mtps_dir=untrained_pot_file_dir.absolute(),
                  wdir=dir.absolute(), 
                  out_name='init.mtp')
    init_path = Path(dir).joinpath('init.mtp')
    cmd = f'{mpirun} {Path(mlip_bin).absolute()} train {init_path.absolute()} {train_set_path.absolute()} {flags}'
    log_path = dir.joinpath('log_train')
    err_path =dir.joinpath('err_train')
    with open(log_path.absolute(), 'w') as log, open(err_path.absolute(), 'w') as err:
        run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
    trained_pot_file_path = dir.joinpath(params['trained_pot_name']).absolute()
    set_level_to_pot_file(trained_pot_file_path=trained_pot_file_path, mtp_level=mtp_level)
    
    if final_evaluation == True:
        eval_dir = dir.joinpath('evaluation')
        if not eval_dir.is_dir():
            eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir.joinpath('ML_dataset.cfg')
        calc_efs(mlip_bin.absolute(),
                 mpirun=mpirun, 
                 confs_path=train_set_path.absolute(),
                 pot_path=trained_pot_file_path,
                 out_path=out_path,
                 dir=eval_dir.absolute())
        
        errs = make_comparison(is_ase1=False,
                        is_ase2=False,
                        structures1=None, 
                        structures2=None, 
                        file1=train_set_path.absolute(),
                        file2=eval_dir.joinpath('ML_dataset.cfg').absolute(),
                        props='all', 
                        make_file=True, 
                        dir=eval_dir,
                        outfile_pref='MLIP-', 
                        units=None)
    if final_evaluation == True:
        return trained_pot_file_path, out_path, errs
    else:
        return trained_pot_file_path
    

def make_mtp_file(sp_count, mind, maxd, rad_bas_sz, rad_bas_type='RBChebyshev', lev=8, mtps_dir=None, wdir='./', out_name='init.mtp'):
    """
    Creates the initial .mtp file necessary to start the MLIP training.

    Args:
        sp_count (int): Number of distinct chemical elements (species_count).
        mind (float): Minimum distance between atoms.
        maxd (float): Cutoff radius for atomic interactions.
        rad_bas_sz (int): Size of the radial basis.
        rad_bas_type (str, optional): Type of radial basis functions. Defaults to 'RBChebyshev'.
        lev (int, optional): MTP level (e.g., 8, 16, 22). Defaults to 8.
        mtps_dir (str or Path, optional): Directory containing the default untrained .mtp files. Defaults to './'.
        wdir (str or Path, optional): Working directory where the new .mtp file will be saved. Defaults to './'.
        out_name (str, optional): Name of the generated .mtp file. Defaults to 'init.mtp'.
    """
    
    if mtps_dir is None:
        mtps_dir = Path('./')
        
    if wdir is not None:
        wdir = Path(wdir)

    lev = int(lev)
    src_name = f'{lev:0>2d}.mtp'
    src_path = mtps_dir.joinpath(src_name)
    
    with open(src_path.absolute(), 'r') as fl:
        lines = fl.readlines()
        
    for i, line in enumerate(lines):
        if 'species_count' in line:
            lines[i] = f'species_count = {sp_count}\n'
        elif 'radial_basis_type' in line:
            lines[i] = f'radial_basis_type = {rad_bas_type}\n'
        elif 'min_dist' in line:
            lines[i] = f'\tmin_dist = {mind}\n'
        elif 'max_dist' in line:
            lines[i] = f'\tmax_dist = {maxd}\n'
        elif 'radial_basis_size' in line:
            lines[i] = f'\tradial_basis_size = {rad_bas_sz}\n'
    outfile_path = wdir.joinpath(out_name)
    text = ''.join(lines)
    with open(outfile_path.absolute(), 'w') as fl:
        fl.write(text)

def set_level_to_pot_file(trained_pot_file_path, mtp_level):
    """
    Updates the potential name tag inside a trained .mtp file to reflect its MTP level.

    Args:
        trained_pot_file_path (str or Path): Path to the trained .mtp file.
        mtp_level (int or float): The MTP level used for training.

    Raises:
        AssertionError: If the file does not exist or if the level is not numeric.
    """
    
    fp = Path(trained_pot_file_path)
    assert fp.is_file(), f"{fp.absolute()} is not a regular file! If you ran a training, check if it has succeeded!"
    assert isinstance(mtp_level, int) or \
           isinstance(mtp_level, float), f"mtp_level must be an integer!"
    mtp_level = int(mtp_level) # in case it's float
    with open(fp.absolute(), 'r') as fl:
        lines = fl.readlines()
    with open(fp.absolute(), 'w') as fl:
        for i, line in enumerate(lines):
            if 'potential_name' in line:
                lines[i] = f'potential_name = MTP_{mtp_level}\n'
        fl.writelines(lines)

def calc_efs(mlip_bin, mpirun='', confs_path='in.cfg', pot_path='pot.mtp', out_path='./out.cfg', dir='./', mute=False):
    """
    Calculates energies, forces, and stresses for configurations in a .cfg file using an MTP.

    Args:
        mlip_bin (str or Path): Path to the MLIP binary.
        mpirun (str, optional): Command for MPI parallelization. Defaults to ''.
        confs_path (str or Path, optional): Path to the input .cfg file. Defaults to 'in.cfg'.
        pot_path (str or Path, optional): Path to the potential .mtp file. Defaults to 'pot.mtp'.
        out_path (str or Path, optional): Path for the output .cfg file. Defaults to './out.cfg'.
        dir (str or Path, optional): Working directory for the calculation. Defaults to './'.
        mute (bool, optional): If True, suppresses writing stdout/stderr to log files. Defaults to False.

    Raises:
        ValueError: If no directory is specified.
    """
    
    if dir is None:
        raise ValueError('Please specify a directory!')
        
    dir = Path(dir)
    if confs_path is not None:
        confs_path = Path(confs_path)
    if pot_path is not None:
        pot_path = Path(pot_path)
    if out_path is not None:
        out_path = Path(out_path)
    
    cmd = f'{mpirun} {mlip_bin.absolute()} calc-efs {pot_path.absolute()} {confs_path.absolute()} {out_path.absolute()}'
    log_path = dir.joinpath('log_calc_efs')
    err_path = dir.joinpath('err_calc_efs')
    if mute is False:
        with open(log_path.absolute(), 'w') as log, open(err_path.absolute(), 'w') as err:
            #print(cmd)
            run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
    else:
        run(cmd.split(), cwd=dir.resolve())

def make_comparison(is_ase1=True,
                    is_ase2=True,
                    structures1=None, 
                    structures2=None, 
                    file1=None,
                    file2=None,
                    props='all', 
                    make_file=False, 
                    dir='./',
                    outfile_pref='', 
                    units=None):
    """
    Creates comparison data and error metrics between true and predicted properties.

    Args:
        is_ase1 (bool, optional): True if the reference structures are ASE objects. Defaults to True.
        is_ase2 (bool, optional): True if the predicted structures are ASE objects. Defaults to True.
        structures1 (ase.Atoms or list, optional): Reference ASE structures. Required if is_ase1=True.
        structures2 (ase.Atoms or list, optional): Predicted ASE structures. Required if is_ase2=True.
        file1 (str or Path, optional): Path to reference .cfg file. Required if is_ase1=False.
        file2 (str or Path, optional): Path to predicted .cfg file. Required if is_ase2=False.
        props (str or list, optional): Properties to compare ('energy', 'forces', 'stress', 'all'). Defaults to 'all'.
        make_file (bool, optional): If True, generates a comparison .dat file. Defaults to False.
        dir (str or Path, optional): Directory to save the comparison file. Defaults to './'.
        outfile_pref (str, optional): Prefix for the output filename, which will be `[outfile_pref][Property]_comparison.dat`. Defaults to ''.
        units (dict, optional): Dictionary mapping properties to their unit strings. Defaults to eV/at, eV/Angstrom, eV/Angstrom^3.

    Returns:
        dict: dictionary whose key/vlaue elements are property/errors, where property can be  {'energy', 'forces', 'stress'}, and errors is a list [rmse, mae, R2].

    Raises:
        ValueError: If an invalid property is requested or structural lengths mismatch.
    """
    
    if is_ase1 == True:
        assert (structures1 != None), f"When is_ase1 = True, " \
            + f"structures1 must be given!"
        if isinstance(structures1, Atoms):
            structures1 = [structures1]
    else:
        assert file1 is not None, f"When is_ase1 = False, file1 must be given!"
        file1 = Path(file1)
        assert file1.is_file() == True, f"{file1.absolute()} is not a file!"
        
    if is_ase2 == True:
        assert (structures2 != None), f"When is_ase2 = True, " \
            + f"structures2 must be given!"
        if isinstance(structures2, Atoms):
            structures2 = [structures2]
    else:
        assert file2 is not None, f"When is_ase2 = False, file2 must be given!"
        file2 = Path(file2)
        assert file2.is_file() == True, f"{file2.absolute()} is not a file!"
    
    if make_file == True:
        dir = Path(dir)
                    
    if isinstance(props, str):
        props = [props]
        
    if not all([x in ['all', 'energy', 'forces', 'stress'] for x in props]):
        raise ValueError("Please give a value or a list of values chosen from ['energy', 'forces', 'stress', 'all']")
    
    if not isinstance(props, list):
        props = [props]
    
    if props == 'all' or (isinstance(props, list) and 'all' in props):
        props =  ['energy', 'forces', 'stress']
    
    if units == None:
        units = dict(energy='eV/at', forces='eV/$\mathrm{\AA}$', stress='eV/$\mathrm{\AA}^2$')
        
    prop_numbs = dict(energy = 0, forces = 1, stress = 2)
    
    # Retrieve the data
    if is_ase1 == True:
        ext1 = [flatten(x) for x in extract_prop_from_ase(structures1)]
    else:
        ext1 = [flatten(x) for x in extract_prop_from_cfg(file1)]

    if is_ase2 == True:
        ext2 = [flatten(x) for x in extract_prop_from_ase(structures2)]
    else:
        ext2 = [flatten(x) for x in extract_prop_from_cfg(file2)]

    if not len(ext1) == len(ext2): # we check here in case structure1 was Atoms and structures2 was cfg. Not ideal, though.
        raise ValueError(f"You gave a different number of true and ML structures!")

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

def extract_prop_from_cfg(filepath):
    """
    Extracts energy, forces, and stress tensors from an MLIP .cfg file.

    Note:
        The convention for the stress tensor in MLIP is: sigma = + dE/dn (strain).
        This function automatically adjusts for the MTP volume multiplication.

    Args:
        filepath (str or Path): Path to the .cfg file.

    Returns:
        tuple: A tuple containing:
            - energy (list of float): Energy PER ATOM for each configuration (eV/atom).
            - forces (list of np.ndarray): Forces array  for each configuration (eV/Angstrom). Shape: (n_confs, n_atoms, 3).
            - stress (list of np.ndarray): Stress tensor for each configuration (eV/Angstrom^3). Shape: (n_confs, 6).
            
    Notes:
        The convention for the stress is that the stress tensor element is: sigma = + dE/dn (n=strain) (note the sign!)
    """
    
    filepath = Path(filepath)
    with open(filepath.absolute(), 'r') as fl:
        lines = fl.readlines()
        nlines = len(lines)
        nconf = 0
        iterator = iter(enumerate(lines))
        forces = []
        energy = []
        stress = []
        
        for i, line in iterator:
            if "BEGIN_CFG" in line:
                nconf += 1
            if "Size" in line:
                natoms = int(lines[i+1])
                curr_forces = np.zeros((natoms, 3))
                curr_stress = np.zeros((6))

            if 'Supercell' in line:
                cell = []
                cell.append( [float(x) for x in lines[i+1].split()] )
                cell.append( [float(x) for x in lines[i+2].split()] )
                cell.append( [float(x) for x in lines[i+3].split()] )
                next(iterator)
                next(iterator)
                next(iterator)
                
            if "AtomData" in line:
                for j in range(natoms):
                    curr_forces[j,0] += float(lines[i + 1 + j].split()[5])
                    curr_forces[j,1] += float(lines[i + 1 + j].split()[6])
                    curr_forces[j,2] += float(lines[i + 1 + j].split()[7])
                forces.append(curr_forces)

                for k in range(natoms):
                    next(iterator)
                    
            if "Energy" in line:
                energy.append(float(lines[i+1])/natoms) # ENERGY PER ATOM!!!!
            
            if "PlusStress" in line:
                V = np.linalg.det(cell)
                curr_stress[0] += (-float(lines[i+1].split()[0])/V) # xx
                curr_stress[1] += (-float(lines[i+1].split()[1])/V) # yy
                curr_stress[2] += (-float(lines[i+1].split()[2])/V) # zz
                curr_stress[3] += (-float(lines[i+1].split()[3])/V) # yz
                curr_stress[4] += (-float(lines[i+1].split()[4])/V) # xz
                curr_stress[5] += (-float(lines[i+1].split()[5])/V) # xy
                stress.append(curr_stress)
        
        return energy, forces, stress

def extract_prop_from_single_cfg(filepath):
    """
    Extracts energy, forces, and stress tensors from an MLIP .cfg file.

    Note:
        The convention for the stress tensor in MLIP is: sigma = + dE/dn (strain).
        This function automatically adjusts for the MTP volume multiplication.

    Args:
        filepath (str or Path): Path to the .cfg file.

    Returns:
        tuple: A tuple containing:
            - energy (float): Energy PER ATOM (eV/atom).
            - forces (np.ndarray): Forces array (eV/Angstrom). Shape: (n_atoms, 3).
            - stress (np.ndarray): Stress tensor (eV/Angstrom^3). Shape: (6).
    """
    
    res = extract_prop_from_cfg(filepath)
    return res[0][0], res[1][0], res[2][0]

def conv_mlip2_to_ase(cfg_path, props=True, elemlist=None):
    """
    Converts an MLIP .cfg structures file back into a list of ASE Atoms objects.

    Note:
        The atomic types in the .cfg file must correspond to the alphabetical 
        order of the chemical symbols provided in the `elemlist`.

    Args:
        cfg_path (str or Path): Path to the .cfg file.
        props (bool, optional): If True, reads properties and stores them in a SinglePointCalculator. Defaults to True.
        elemlist (list): List of chemical symbols present in the system to map integer types back to elements.

    Returns:
        list of ase.Atoms: Trajectory of ASE configurations.
    """

    with open(cfg_path, 'r') as fl:
        lines = fl.readlines()
    skip_counting = 0
    confs = []
    elemlist = [x for x in set(elemlist)]
    elemlist.sort()
    elems = dict()
    for j, el in enumerate(elemlist):
        elems[j] = el

    for i, line in enumerate(lines):
        if skip_counting > 0:
            skip_counting = skip_counting - 1
            continue
        elif 'BEGIN' in line:
            indices = []
            positions = []
            forces = []
            stress = []
        elif 'Size' in line:
            size = int(lines[i+1])
        elif 'Supercell' in line:
            cell = []
            cell.append([float(x) for x in lines[i+1].split()])
            cell.append([float(x) for x in lines[i+2].split()])
            cell.append([float(x) for x in lines[i+3].split()])
            cell = np.array(cell)
            volume = np.linalg.det(cell)
            skip_counting = 3
        elif 'AtomData' in line:
            for j in range(1, size+1):
                data = lines[i+j].split()
                indices.append(int(data[1]))
                position = [float(data[2]), float(data[3]), float(data[4])]
                positions.append(position)
                if props == True:
                    force = [float(data[5]), float(data[6]), float(data[7])]
                    forces.append(force)
            symbols = [elems[x] for x in indices]
            skip_counting = size
            
        elif props == True and 'Energy' in line:
            energy = float(lines[i+1])
        elif props == True and 'PlusStress' in line:
            data = [float(x) for x in lines[i+1].split()]
            stress = -np.array(data)/volume
        
        elif 'END_CFG' in line:
            conf = Atoms(symbols=symbols, positions=positions, cell=cell)
            if props == True:
                calc = SinglePointCalculator(atoms=conf, energy=energy, forces=forces, stress=stress)
                conf.calc = (calc)
                conf.get_potential_energy()
            confs.append(conf)
    return confs

def calc_efs_from_single_ase(mlip_bin,
                      atoms,
                      mpirun='',
                      pot_path='pot.mtp',
                      cfg_files=False,
                      out_path='./out.cfg',
                      dir='./',
                      write_conf=False,
                      outconf_name=None,
                      mute=False):
    """
    Calculates and assigns energies, forces, and stresses for a single ASE Atoms object using an MTP.

    Args:
        mlip_bin (str or Path): Path to the MLIP binary.
        atoms (ase.Atoms): The single atomic configuration to evaluate.
        mpirun (str, optional): Command for MPI parallelization. Defaults to ''.
        pot_path (str or Path, optional): Path to the potential .mtp file. Defaults to 'pot.mtp'.
        cfg_files (bool, optional): If True, keeps the intermediate .cfg files generated. Defaults to False.
        out_path (str or Path, optional): Filename for the output .cfg file. Defaults to './out.cfg'.
        dir (str or Path, optional): Working directory. Defaults to './'.
        write_conf (bool, optional): If True, saves the evaluated Atoms object to a trajectory file. Defaults to False.
        outconf_name (str, optional): Filename for the saved trajectory. Defaults to 'confs.traj'.
        mute (bool, optional): If True, suppresses writing stdout/stderr to log files. Defaults to False.

    Returns:
        ase.Atoms: The input Atoms object with calculated properties stored in its calculator.
    """
    
    if dir is None:
        raise ValueError('Please specify a directory!')

    dir = Path(dir)
    if pot_path is not None:
        pot_path = Path(pot_path)
    if out_path is not None:
        out_path = Path(out_path)
    if mlip_bin is not None:
        mlip_bin = Path(mlip_bin)

    # first we need to convert ASE to cfg
    cfg_traj = dir.joinpath('in.cfg')
    conv_single_ase_to_mlip2(atoms, cfg_traj.absolute(), props=False)

    # compute the properties
    calc_efs(mlip_bin.absolute(), mpirun=mpirun, confs_path=cfg_traj.absolute(), pot_path=pot_path.absolute(), out_path=dir.joinpath(out_path).absolute(), dir=dir.absolute(), mute=mute)

    # extract the properties from the results
    energy, forces, stress = extract_prop_from_single_cfg(filepath=dir.joinpath(out_path).absolute()) # energy per atom!!
    # create the SinglePoint calculator and assign it to the structure, then "compute" the properties
    calc = SinglePointCalculator(atoms, energy=energy*len(atoms), forces=forces, stress=stress)
    atoms.calc = calc
    atoms.get_potential_energy()

    if write_conf == True:
        if outconf_name is None:
            outconf_name = f'confs.traj'
        write(dir.joinpath(outconf_name).absolute(), atoms)

    if cfg_files == False:
        cfg_traj.unlink(missing_ok=True)
        dir.joinpath(out_path).unlink(missing_ok=True)
    return atoms

def calc_efs_from_ase(mlip_bin, 
                      atoms, 
                      mpirun='', 
                      pot_path='pot.mtp', 
                      cfg_files=False, 
                      out_path='./out.cfg',
                      dir='./',
                      write_conf=False, 
                      outconf_name=None,
                      mute=False):
    """
    Calculates and assigns energies, forces, and stresses for an ASE trajectory using an MTP.

    Args:
        mlip_bin (str or Path): Path to the MLIP binary.
        atoms (list of ase.Atoms or ase.Atoms): The trajectory or single configuration to evaluate.
        mpirun (str, optional): Command for MPI parallelization. Defaults to ''.
        pot_path (str or Path, optional): Path to the potential .mtp file. Defaults to 'pot.mtp'.
        cfg_files (bool, optional): If True, keeps the intermediate .cfg files generated. Defaults to False.
        out_path (str or Path, optional): Filename for the output .cfg file. Defaults to './out.cfg'.
        dir (str or Path, optional): Working directory. Defaults to './'.
        write_conf (bool, optional): If True, saves the evaluated trajectory to a file. Defaults to False.
        outconf_name (str, optional): Filename for the saved trajectory. Defaults to 'confs.traj'.
        mute (bool, optional): If True, suppresses writing stdout/stderr to log files. Defaults to False.

    Returns:
        list of ase.Atoms: The input configurations with calculated properties stored in their calculators.
    """
    
    if dir is None:
        raise ValueError('Please specify a directory!')
        
    dir = Path(dir)
    if pot_path is not None:
        pot_path = Path(pot_path)
    if out_path is not None:
        out_path = Path(out_path)
    if mlip_bin is not None:
        mlip_bin = Path(mlip_bin)
    
    # first we need to convert ASE to cfg
    cfg_traj = dir.joinpath('in.cfg')
    conv_ase_to_mlip2(atoms, cfg_traj.absolute(), props=False)
    
    # compute the properties
    calc_efs(mlip_bin.absolute(), mpirun=mpirun, confs_path=cfg_traj.absolute(), pot_path=pot_path.absolute(), out_path=dir.joinpath(out_path).absolute(), dir=dir.absolute(), mute=mute)
    
    # extract the properties from the results
    energy, forces, stress = extract_prop(filepath=dir.joinpath(out_path).absolute()) # energy per atom!!
    # for each configuration create the SinglePoint calculator and assign it, then "compute" the properties
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    for i, atom in enumerate(atoms):
        calc = SinglePointCalculator(atom, energy=energy[i]*len(atom), forces=forces[i], stress=stress[i])
        atom.calc = calc
        atom.get_potential_energy()
    
    if write_conf == True:
        if outconf_name is None:
            outconf_name = f'confs.traj'
        write(dir.joinpath(outconf_name).absolute(), atoms)
    
    if cfg_files == False:
        cfg_traj.unlink(missing_ok=True)
        dir.joinpath(out_path).unlink(missing_ok=True)
    return atoms

def extract_prop(structures=None, filepath=None):
    """
    Universal property extractor that handles both ASE Atoms objects and MLIP .cfg files.

    Note:
        Only one of `structures` or `filepath` can be provided. 
        Stress tensor convention is: sigma = + dE/dn (strain).

    Args:
        structures (ase.Atoms or list of ase.Atoms, optional): Configurations to extract from.
        filepath (str or Path, optional): Path to a .cfg file to extract from.

    Returns:
        tuple: A tuple containing:
            - energy (np.ndarray): Energy per atom for each configuration (eV/atom).
            - forces (np.ndarray): Forces array (confs x atoms x 3) (eV/Angstrom).
            - stress (np.ndarray): Stress tensor (confs x 6) (eV/Angstrom^3).

    Raises:
        AssertionError: If both or neither `structures` and `filepath` are provided.
        TypeError: If `structures` contains invalid types.
    """
    
    if filepath is not None:
        filepath = Path(filepath)
    assert any([structures != None, filepath != None]), f"Either structure or filepath must be given!"
    assert not all([structures != None, filepath != None]), f"Either structure or filepath can be given!"
    if structures != None:
        if isinstance(structures, Atoms):
            structures = [structures]
        elif isinstance(structures, list):
            assert all([isinstance(x, Atoms) for x in structures]), \
                   f"Some element of structures is not an ase.atoms.Atoms object!"
        else: 
            raise TypeError('The structures argument passed must be either an Atom object (or a list of Atom objects) or a .cfg file!')
        return extract_prop_from_ase(structures)
    else:
        return extract_prop_from_cfg(filepath.absolute())
    
def make_ini_for_lammps(pot_file_path, out_file_path):
    """
    Generates a LAMMPS initialization (.ini) file for MTP potentials.

    This file is required by the MLIP-LAMMPS interface to link the MLIP 
    binary logic with a specific potential file.

    Args:
        pot_file_path (str or Path): Path to the trained .mtp potential file.
        out_file_path (str or Path): Destination path where the .ini file will be created.

    Returns:
        None

    Note:
        The output directory is created automatically if it does not exist. 
        The 'select' flag is hardcoded to FALSE.
    """
    
    pot_file_path = Path(pot_file_path)
    txt = f'mtp-filename {pot_file_path.absolute()}\n'
    txt += f"select FALSE"
    out_file_path = Path(out_file_path)
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_path.absolute(), 'w') as fl:
        fl.writelines(txt)

def find_min_dist(trajectory):
    """
    Finds the absolute minimum interatomic distance across an entire trajectory.

    This is a critical utility for setting the 'min_dist' hyperparameter in 
    potentials like MTP to avoid unphysical overlaps during training.

    Args:
        trajectory (ase.Atoms or list of ase.Atoms): A single configuration or a 
            sequence of ASE Atoms objects.

    Returns:
        float: The smallest pairwise distance (in Angstrom) found across all atoms 
            and all frames, considering periodic boundary conditions (MIC).

    Raises:
        ValueError: If the provided trajectory is empty.
    """
    
    if not trajectory:
        raise ValueError("Trajectory is empty.")
    
    if isinstance(trajectory, Atoms):
        trajectory = [trajectory]

    global_min = np.inf
    for conf in trajectory:
        dist_matrix = conf.get_all_distances(mic=True)
        np.fill_diagonal(dist_matrix, np.inf)
        global_min = min(global_min, dist_matrix.min())
    return global_min

class LAMMPS_custom(LAMMPS):
    """
    Custom wrapper for the ASE LAMMPS calculator to enhance directory tracking.

    This subclass extends the standard LAMMPS calculator to provide explicit 
    feedback on temporary directory locations during the calculation process.

    Methods:
        calculate(atoms, properties, system_changes): Executes the LAMMPS 
            calculation and prints the resolved path of the working directory.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calculate(self, *args, **kwargs):
        super().calculate(*args, **kwargs)
        self.parameters['tmp_dir']
        print(Path(self.parameters['tmp_dir']).resolve())
         
