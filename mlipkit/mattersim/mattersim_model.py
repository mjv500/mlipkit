import numpy as np
import os
import sys
import shutil
import random

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ..mlip_models import MlipModel
from .mattersim_utils import finetune_pot
from mattersim.forcefield import MatterSimCalculator


class MatterSimModel(MlipModel):
    class_obj_name = 'mattersim_model' # the name that will be used to save the object

    mandatory_hyperparameters_names = []
    optional_hyperparameters_names = ['device',
                                      'run_name',
                                      'finetune_script_path',
                                      'validation_fraction',
                                      'shuffle_dataset',
                                      'clean_after_training',
                                      'load_model_path',
                                      'ckpt_interval',
                                      'cutoff',
                                      'threebody_cutoff',
                                      'epochs',
                                      'batch_size',
                                      'learning_rate',
                                      'step_size',
                                      'include_forces',
                                      'include_stresses',
                                      'force_loss_ratio',
                                      'stress_loss_ratio',
                                      'early_stop_patience',
                                      'seed',
                                      're_normalize',
                                      'scale_key',
                                      'shift_key',
                                      'init_scale',
                                      'init_shift',
                                      'trainable_scale',
                                      'trainable_shift',
                                      'wandb',
                                      'wandb_api_key',
                                      'wandb_project',
                                      'set_trained_pot_as_new_starting_point']

    
    mandatory_parameters_compute_properties_names = []
    optional_parameters_compute_properties_names = ['device']

    computable_properties_names = ['energy', 'forces', 'stress']

    trained_pot_filename = 'pot.pth' # name of the .mtp file that will be produced by the training    
    trained_pot_files = {'potential_file' : 'pot.pth'} # key: generic name of the file, value: actual filename

    trainable = True

    # minimum size of training set
    min_train_set_size = 2

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
        train_set: list, ase.atoms.Atoms 
            list of ase Atoms objects; energy, forces and stresses must have been 
            computed and stored in each Atoms object        
        ''' 
        super().__init__(name,
                         root_dir, 
                         hyperparameters=hyperparameters,
                         parameters_compute_properties=parameters_compute_properties,
                         train_set=train_set, 
                         save_model=save_model,
                         pre_trained=pre_trained,
                         pre_trained_pot_filepaths=pre_trained_pot_filepaths)

    def _train(self):
        """
        Executes the MatterSim fine-tuning process using the provided training set.

        This method prepares the dataset by splitting off a validation set, 
        optionally shuffling the configurations, and calling the external 
        fine-tuning engine. It manages the promotion of the 'best' model 
        to the project directory and performs optional cleanup of heavy 
        checkpoint and trajectory files.

        Returns:
            bool: True if training succeeded and 'best_model.pth' was produced; 
                False otherwise.

        Note:
            If 'set_trained_pot_as_new_starting_point' is True, the internal 
            hyperparameters are updated to point to the newly trained weights 
            for future training sessions.
        """
        
        # prepare the dictionary from self.hyperparameters with ONLY existing parameters (so the non-existent ones will be set to default by finetune_pot();
        # note that setting them = None would not trigger the default value)
        self.hyperparameters['save_checkpoint'] = True # if False, it won't print the trained pot file
        self.hyperparameters.setdefault('set_trained_pot_as_new_starting_point', False)
        self.hyperparameters.setdefault('validation_fraction', 0.2)
        self.hyperparameters.setdefault('clean_after_training', True)
        self.hyperparameters.setdefault('shuffle_dataset', True)
        validation_fraction = self.hyperparameters['validation_fraction'] 
        if validation_fraction <= 0:
            raise ValueError('Please give a fraction for the validation set > 0!')
        n_validation_confs = int(len(self.train_set) *  validation_fraction)
        if n_validation_confs == 0:
            n_validation_confs = 1
        #train_set_copy = deepcopy(train_set_copy)
        
        indices = list(range(len(self.train_set)))

        args_to_pass = {}
        for key in self.all_hyperparameters_names:
            if key in self.hyperparameters:
                args_to_pass[key] = self.hyperparameters[key]

        if self.hyperparameters['shuffle_dataset'] is True:
            tmp_dataset = random.sample(self.train_set, len(self.train_set))
        else:
            tmp_dataset = self.train_set
        trained_model_path = finetune_pot(wdir=self.get_training_dir(),
                                          train_set = tmp_dataset[:-n_validation_confs],
                                          test_set = tmp_dataset[-n_validation_confs:],
                                          final_evaluation=False,
                                          eval_dir=None,
                                          saving_path=self.get_training_dir(),
                                          delete_sets=self.hyperparameters['clean_after_training'],
                                          **args_to_pass)
        # after this, we take the best model and save it inside the training_directory
        src_pot = trained_model_path.resolve()
        target_pot = self.get_training_dir().joinpath(self.trained_pot_files['potential_file']).resolve() # save it inside training_dir; it will be copied into trained_pot_dir later
        shutil.copy(src_pot, target_pot) # this is the name that will be looked for after the training, to store the trained model in the trained_pot_dir
        # clean
        if self.hyperparameters['clean_after_training'] == True:
            [x.unlink(missing_ok=True) for x in self.get_training_dir().glob('ckpt*')]
            self.get_training_dir().joinpath('best_model.pth').unlink(missing_ok=True)
            self.get_training_dir().joinpath('Train_set.traj').unlink(missing_ok=True)
            self.get_training_dir().joinpath('Test_set.traj').unlink(missing_ok=True)
            
        # must return True for success and False otherwise
        if self.get_training_dir().joinpath('last_model.pth').is_file():
            success = True
        else:
            success = False

        if success == True:
            if self.hyperparameters['set_trained_pot_as_new_starting_point'] == True:
                self.hyperparameters['load_model_path'] = self.get_trained_pot_dir().joinpath(self.trained_pot_files['potential_file']).resolve()
        return success

    def _compute_properties_single(self, atoms, wdir, parameters):
        """
        Calculates properties for a single configuration using MatterSim.

        Args:
            atoms (ase.Atoms): The structure to evaluate.
            wdir (str or Path): Ignored in this implementation as MatterSim 
                operates in-memory.
            parameters (dict): Compute parameters, including 'device' (cpu/cuda).

        Returns:
            ase.Atoms: The configuration with results stored in a persistent 
                SinglePointCalculator.
        """
        
        # we don't need wdir in this case; we'll ignore it
        # parameters
        if 'device' in parameters.keys():
            device = parameters['device']
        else:
            device = 'cpu'
        
        pot_path = self.get_pot_filepath('potential_file')

        mattersim_calc = MatterSimCalculator(load_path=str(pot_path), device=device)
         
        atoms.calc = mattersim_calc
        atoms.get_total_energy()
        energy = atoms.calc.results['energy'] 
        forces = atoms.calc.results['forces']
        stress = atoms.calc.results['stress']
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        atoms.calc = calc
        return atoms
    
    def _compute_properties_list(self, atoms, wdir, parameters):
        """
        Calculates properties for a list of configurations via sequential evaluation.

        Args:
            atoms (list of ase.Atoms): Configurations to evaluate.
            wdir (str or Path): Ignored.
            parameters (dict): Compute parameters (e.g., 'device').

        Returns:
            list of ase.Atoms: Evaluated structures with persistent results.
        """
        
        # we don't need wdir in this case; we'll ignore it
        # parameters
        
        atoms = [self._compute_properties_single(atom, None, parameters) for atom in atoms]
        return atoms
    
    def make_ase_calculator(self, parameters=None):
        """
        Initializes and returns a persistent MatterSim ASE calculator.

        This allows the model to be used directly in standard ASE workflows 
        (e.g., MD, geometry optimization) using the trained potential.

        Args:
            parameters (dict, optional): Compute parameters to set or override.

        Returns:
            MatterSimCalculator: The initialized ASE-compatible calculator object.

        Raises:
            KeyError: If mandatory compute parameters are missing.
        """
        
        if parameters is not None:
            res = self._check_mandatory_parameters_compute_properties(parameters)
        else:
            res = self._check_mandatory_parameters_compute_properties()
        
        if res is not True: # then `res` is the name of the missing mandatory parameter
            raise KeyError(f'The {type(self).class_obj_name} object has no parameter called `{res}` for computing properties, and you didn\'t provide any!')
        if parameters is not None:
            self.parameters_compute_properties = parameters
            
        if 'device' in self.parameters_compute_properties.keys():
            device = self.parameters_compute_properties['device']
        else:
            device = 'cpu'
        load_path=str(self.get_pot_filepath('potential_file'))
        self.ase_calculator = MatterSimCalculator(load_path=load_path, device=device)
        return self.ase_calculator
        




