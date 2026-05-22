#check
import numpy as np
from copy import deepcopy
from pathlib import Path
import shutil
import os

from ase.io import read, write
from ase.atoms import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.singlepoint import SinglePointCalculator

from .mtp_utils import train_pot_from_ase, calc_efs_from_single_ase, make_ini_for_lammps
from ..mlip_models import MlipModel
from ..utils import flatten




class MTPModel(MlipModel):
    '''Implementation of Moment Tensor Potential as MlipModel subclass'''

    class_obj_name = 'MTP_model' # the name that will be used to save the object

    # training hyperparameters
    mandatory_hyperparameters_names = ['mlip_bin',
                                       'untrained_pot_file_dir',
                                       'mtp_level',
                                       'min_dist',
                                       'max_dist',
                                       'radial_basis_type',
                                       'radial_basis_size']
    
    optional_hyperparameters_names = ['ene_weight', 
                                      'for_weight',
                                      'str_weight',
                                      'sc_b_for',
                                      'val_cfg',
                                      'max_iter',
                                      'cur_pot_n', 
                                      'bfgs_tol', 
                                      'weighting', 
                                      'init_par', 
                                      'skip_preinit', 
                                      'up_mindist',
                                      'bin_pref']
    
    # parameters to predict properties
    mandatory_parameters_compute_properties_names = ['lammps_bin']
    #mandatory_parameters_compute_properties_names = ['mlip_bin']

    optional_parameters_compute_properties_names = ['bin_pref']

    # implemented properties
    computable_properties_names = ['energy',
                                   'forces',
                                   'stress']
    
    # trained potential files
    trained_pot_filename = 'pot.mtp' # name of the .mtp file that will be produced by the training    
    trained_pot_files = {'potential_file' : 'pot.mtp'} # key: generic name of the file, value: actual filename

    # trainable
    trainable = True
    
    # minimum size of training set
    min_train_set_size = 1

    def __init__(self,
                 root_dir='./',
                 name = None,
                 hyperparameters=None,
                 parameters_compute_properties=None,
                 train_set=None,
                 save_model=False,
                 pre_trained=False,
                 pre_trained_pot_filepaths=None):
        '''
        Args:
            root_dir (str or Path): root directory.
            name (str or Path): custom name for the object.
            hyperparameters (dict): dictionary with the training hyperparameters {hyperpar_name : hyperpar_value}.
            train_set (list, ase.atoms.Atoms): training set.
            save_model (bool): True: save the model as JSON right after the initialization.
            pre_trained (bool): True if trying to load a pre-trained model. 
            pre_trained_pot_filepaths (dict): dictionary with the filepaths needed to define the trained potential {generic_name : filepath}
                (`generic_name` is hardcoded, you can check its values with MTP_model.load_doc('trained_pot_files')).
            save_train_set : 
            train_set_saving_loc :         
        ''' 
        super().__init__(name,
                         root_dir, 
                         hyperparameters=hyperparameters,
                         parameters_compute_properties = parameters_compute_properties,
                         train_set=train_set,
                         save_model=save_model,
                         pre_trained=pre_trained,
                         pre_trained_pot_filepaths=pre_trained_pot_filepaths)

    def _train(self):
        """
        Executes the Moment Tensor Potential (MTP) training via the mlip-2 binary.

        This method prepares the command-line parameters from the model's 
        hyperparameters, calls the external training engine, and validates 
        the completion of the process by inspecting the 'log_train' file.

        Returns:
            bool: True if training completed successfully (found error reports 
                in log), False otherwise.

        Raises:
            FileNotFoundError: If the 'log_train' file is not generated.
        """
        
        keys_for_params = ['ene_weight',
                           'for_weight',
                           'str_weight',
                           'sc_b_for',
                           'val_cfg',
                           'max_iter',
                           'cur_pot_n',
                           'bfgs_tol',
                           'weighting',
                           'init_par',
                           'skip_preinit',
                           'up_mindist']

        existing_keys = self.hyperparameters.keys() # list of the parameters possessed by self;
        params = {key: self.hyperparameters[key] for key in keys_for_params if key in existing_keys} # not all keys_for_params are mandatory, so we check
        params['trained_pot_name'] = 'pot.mtp' # hardcoded
        if 'bin_pref' in existing_keys:
            bin_pref = self.hyperparameters['bin_pref']
        else:
            bin_pref = ''
        train_pot_from_ase(mlip_bin=self.hyperparameters['mlip_bin'],
                           untrained_pot_file_dir=self.hyperparameters['untrained_pot_file_dir'],
                           mtp_level=self.hyperparameters['mtp_level'],
                           min_dist=self.hyperparameters['min_dist'],
                           max_dist=self.hyperparameters['max_dist'],
                           radial_basis_size=self.hyperparameters['radial_basis_size'],
                           radial_basis_type=self.hyperparameters['radial_basis_type'],
                           train_set=self.train_set,
                           dir=self.get_training_dir(),
                           params=params,
                           mpirun=bin_pref,
                           final_evaluation=False)
        
        # we need to check that the training went to completion
        with open(self.get_training_dir().joinpath('log_train'), 'r') as fl:
            lines = fl.readlines()
            if any(['* * * TRAIN ERRORS * * *' in line for line in lines]):
                return True 
            else:
                return False  

    def old_compute_properties_list(self, atoms, wdir, parameters):
        '''
        atoms: list of Atoms objects

        Return
        atoms_calc: of Atoms objects
        '''
        # wdir
        wdir = Path(wdir)
        wdir.mkdir(exist_ok=True, parents=True)
        # other parameters
        mlip_bin = parameters['mlip_bin']
        if 'bin_pref' in parameters.keys():
            bin_pref = parameters['bin_pref']
        else:
            bin_pref = ''
        atoms_calc = calc_efs_from_ase(mlip_bin=mlip_bin, 
                                            atoms=atoms, 
                                            mpirun=bin_pref, 
                                            pot_path=self.get_pot_filepath('potential_file'), 
                                            cfg_files=False, 
                                            dir=wdir,
                                            write_conf=False,
                                            mute=parameters['mute'])
        return atoms_calc

    def old_compute_properties_single(self, atoms, wdir, parameters):
        '''
        atoms: an Atoms object (not a list!!)

        Return:
        atoms_calc: an Atoms object
        '''
        # wdir
        wdir = Path(wdir)
        wdir.mkdir(exist_ok=True, parents=True)
        # other parameters
        mlip_bin = parameters['mlip_bin']
        if 'bin_pref' in parameters.keys():
            bin_pref = parameters['bin_pref']
        else:
            bin_pref = ''
        atoms_calc = calc_efs_from_single_ase(mlip_bin=mlip_bin,
                                            atoms=atoms,
                                            mpirun=bin_pref,
                                            pot_path=self.get_pot_filepath('potential_file').resolve(),
                                            cfg_files=False,
                                            dir=wdir,
                                            write_conf=False,
                                            mute=parameters['mute'])
        return atoms_calc

    def _compute_properties_single(self, atoms, wdir, parameters):
        """
        Computes properties for a single Atoms object using the LAMMPS calculator.

        Args:
            atoms (ase.Atoms): The configuration to evaluate.
            wdir (str or Path): Working directory for the LAMMPS run.
            parameters (dict): Parameters containing 'bin_pref' and 'lammps_bin'.

        Returns:
            ase.Atoms: The Atoms object with calculated energy, forces, and stress.

        Note:
            This method sets the 'ASE_LAMMPSRUN_COMMAND' environment variable 
            and generates a temporary 'mlip.ini' file in the wdir.
        """
        
        wdir = Path(wdir)
        wdir.mkdir(exist_ok=True, parents=True)
        pot_path = self.get_pot_filepath('potential_file')
        make_ini_for_lammps(pot_file_path=pot_path, out_file_path=wdir.joinpath('mlip.ini'))
        pair_style = f"mlip {wdir.joinpath('mlip.ini').absolute()}"
        pair_coeff = ["* *"]
        os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{parameters["bin_pref"]} {parameters["lammps_bin"]}'
        calc = LAMMPS(pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    wdir='WTF_ASE')
        atoms.calc = calc
        atoms.get_potential_energy()
        return atoms

    def _compute_properties_list(self, atoms, wdir, parameters):
        """
        Computes properties for a list of Atoms objects using a looped LAMMPS evaluation.

        Each configuration is evaluated via a LAMMPS calculator, and the 
        results are stored back into the Atoms objects via a SinglePointCalculator 
        to ensure persistence.

        Args:
            atoms (list of ase.Atoms): Configurations to evaluate.
            wdir (str or Path): Working directory for temporary files.
            parameters (dict): Calculation parameters (binaries and prefixes).

        Returns:
            list of ase.Atoms: Evaluated configurations with updated calculators.
        """
        
        wdir = Path(wdir)
        wdir.mkdir(exist_ok=True, parents=True)
        pot_path = self.get_pot_filepath('potential_file')
        make_ini_for_lammps(pot_file_path=pot_path, out_file_path=wdir.joinpath('mlip.ini'))
        pair_style = f"mlip {wdir.joinpath('mlip.ini').absolute()}"
        pair_coeff = ["* *"]
        os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{parameters["bin_pref"]} {parameters["lammps_bin"]}'
        calc = LAMMPS(pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    wdir='WTF_ASE')
        for atom in atoms:
            atom.calc = calc
            atom.get_potential_energy()
            energy = atom.calc.results['energy']
            forces = atom.calc.results['forces']
            stress = atom.calc.results['stress']
            new_calc = SinglePointCalculator(atoms=atom, energy=energy, forces=forces, stress=stress)
            atom.calc = new_calc
        return atoms
    
                
    def make_ase_calculator(self, parameters=None):
        """
        Initializes and attaches a persistent ASE LAMMPS calculator to the model.

        This method configures the environment and the LAMMPS pair_style for 
        MLIP, allowing the model to be used as a standard ASE calculator 
        object (`self.ase_calculator`).

        Args:
            parameters (dict, optional): Parameters to override or set the 
                compute configuration. Must contain 'lammps_bin'.

        Returns:
            None

        Raises:
            KeyError: If mandatory compute parameters (like 'lammps_bin') are missing.
        """
        
        if parameters is not None:
            res = self._check_mandatory_parameters_compute_properties(parameters)
        else:
            res = self._check_mandatory_parameters_compute_properties()
        
        if res is not True: # then `res` is the name of the missing mandatory parameter
            raise KeyError(f'The {type(self).class_obj_name} object has no parameter called `{res}` for computing properties, and you didn\'t provide any!')
        if parameters is not None:
            self.parameters_compute_properties = parameters
            
        pot_file_path = self.get_pot_filepath('potential_file')
        out_file_path = self.root_dir.joinpath('mlip.ini')
        make_ini_for_lammps(pot_file_path=pot_file_path, out_file_path=out_file_path)
        self.parameters_compute_properties.setdefault('bin_pref', '')
        lammps_cmd = f"{self.parameters_compute_properties['bin_pref']} {self.parameters_compute_properties['lammps_bin']}"
        os.environ["ASE_LAMMPSRUN_COMMAND"] = lammps_cmd
        self.get_tmp_dir().mkdir(parents=True, exist_ok=True)
        print(f"wdir = {str(self.get_tmp_dir())}")
        lammps_calc = LAMMPS(pair_style=f"mlip {out_file_path.resolve()}",
                             pair_coeff=['* *'],
                             tmp_dir=str(self.get_tmp_dir()),
                             keep_tmp_files=False)
        self.ase_calculator = lammps_calc


