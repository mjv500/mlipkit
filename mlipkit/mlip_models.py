### check
from abc import ABC, abstractmethod
import numpy as np
import json
from pathlib import Path
import shutil
import traceback
import importlib
import sys
from copy import deepcopy

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator 
from ase.io import read, write
from ase.data import chemical_symbols

from .utils import hash_multiple_files, atoms_hash, flatten
from .mlip_utils import make_comparison
from .mlipkit_calculator import MlipkitCalculator




class MlipModel(ABC):

    class_obj_name = 'MlipModel'
    _subclass_registry = {}

    mandatory_hyperparameters_names = []
    optional_hyperparameters_names = []

    mandatory_parameters_compute_properties_names = []
    optional_parameters_compute_properties_names = []

    computable_properties_names = []

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the model registry."""        
        super().__init_subclass__(**kwargs) # not really necessary in this case
        full_path = f"{cls.__module__}.{cls.__name__}"
        MlipModel._subclass_registry[cls.__name__] = full_path
    
    @classmethod
    def _get_class_from_string(cls, class_path):
        """Load a class from a string path.

        Args:
            class_path (str): Full module path of the class (e.g., 'package.module.ClassName').

        Returns:
            type: The class object referenced by the string.
        """
        module_path, class_name = class_path.rsplit('.', 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    

    @classmethod
    def load_doc(cls, key=None):
        """
        Load the documentation JSON file for the model class.

        Args:
            key (str, optional): If provided, only return the corresponding section 
                                 (e.g., 'hyperparameters').

        Returns:
            dict or any: Full documentation dictionary or sub-section if `key` is given.

        Raises:
            FileNotFoundError: If the documentation file does not exist.
            KeyError: If the provided key is not in the documentation.
        """
        
        # This finds the directory where the .py file of the SUBCLASS is located
        current_module = sys.modules[cls.__module__]
        if hasattr(current_module, '__file__'):
            doc_path = Path(current_module.__file__).parent / 'parameters_doc.json'
        else:
            # Fallback for interactive environments
            raise FileNotFoundError("Could not determine module path for documentation.")
        
        with open(doc_path, 'r') as f:
            doc = json.load(f)

        if key is not None:
            if key not in doc:
                raise KeyError(f"Key '{key}' not found in documentation.")
            else:
                for k, v in doc[key].items():
                    print(f'{k}: {v}\n')
        else:
            for key in doc.keys():
                print(f'{key}:')
                for k, v in doc[key].items():
                    print(f'{k}: {v}\n')

    # def _get_class_from_string(class_path):
    #     module_path, class_name = class_path.rsplit('.', 1)
    #     mod = importlib.import_module(module_path)
    #     return getattr(mod, class_name)

    def __init__(self, 
                 name,
                 root_dir='./', 
                 train_set=None, 
                 hyperparameters=None, 
                 parameters_compute_properties=None,
                 save_model=False,
                 pre_trained=False, 
                 pre_trained_pot_filepaths=None):
        
        if name is None:
            self.name = type(self).class_obj_name
        else:
            self.name = name
        self.root_dir = Path(root_dir)
        self.tmp_dirname = 'mlipkit_tmp'
        self.tmp_dir = self.root_dir.joinpath(self.tmp_dirname) # we don't create it yet. Everytime it is used, it must be created
        if pre_trained == False:
            self.root_dir.mkdir(exist_ok=True, parents=True)
        self.training_dirname = 'training'
        self.training_dir = self.root_dir.joinpath(self.training_dirname) 
        self.trained_pot_dirname = 'trained_pot'
        self.trained_pot_dir = self.root_dir.joinpath(self.trained_pot_dirname)
        #self.mkdir(exist_ok=True, parents=True)


        self.mandatory_hyperparameters_names = type(self).mandatory_hyperparameters_names
        self.optional_hyperparameters_names = type(self).optional_hyperparameters_names
        self.all_hyperparameters_names = self.mandatory_hyperparameters_names + self.optional_hyperparameters_names

        self.mandatory_parameters_compute_properties_names = type(self).mandatory_parameters_compute_properties_names
        self.optional_parameters_compute_properties_names = type(self).optional_parameters_compute_properties_names
        self.all_parameters_compute_properties_names = self.mandatory_parameters_compute_properties_names + self.optional_parameters_compute_properties_names
        
        self.computable_properties_names = type(self).computable_properties_names

        self.is_trainable = type(self).trainable
        #self.is_finetunable = type(self).is_finetunable  to be implemented
        
        self.min_train_set_size = type(self).min_train_set_size

        self.trained_pot_files = type(self).trained_pot_files
        #self.trained_pot_filepaths = {k: self.trained_pot_dir.joinpath(v).resolve() for k, v in self.trained_pot_files.items()}
        #self.trained_pot_filepaths = {k: None for k, v in self.trained_pot_files.items()} # we initialize each file with None

        self.train_set = train_set
        if self.train_set is not None:
            # Get unique atomic numbers across the whole dataset
            all_numbers = set()
            for atoms in self.train_set:
                all_numbers.update(atoms.get_atomic_numbers())
            self.species_count = sorted([chemical_symbols[n] for n in all_numbers])
            #self.species_count = list(set(flatten([x.get_chemical_symbols() for x in self.train_set])))
        else:
            self.species_count = None

        self.ml_trainset = None # we need to initialize it
        self.trainset_bck = None # we need to initialize it
        
        
        self.is_trained = False
        self.regular = dict(d=False, h=False) # d = dataset, h = hyperparameters
        self._is_pre_trained = False

        self.hash_train_set = None
        self.hash_trained_pot = None

        # now if the hyperparameters were given, we need to check that only the proper ones are in the list.
        # We don't check that all the mandatory ones are included, for the time being. 
        # It will be done when trying to train or a bit later if this is the loading of a pre-trained model 
        
        self.hyperparameters = dict()
        if hyperparameters is not None:
            for hp_n, hp_v in hyperparameters.items():
                if hp_n in self.all_hyperparameters_names:
                    self.hyperparameters[hp_n] = hp_v

        # if the compute_props parameters are given, we need to check that only the proper ones are in the list.
        # We don't need to check that all the mandatory ones are included, for the time being.
        # It will be done when (if) trying to compute the properties
        self.parameters_compute_properties = dict()
        if parameters_compute_properties is not None:
            for p_n, p_v in parameters_compute_properties.items():
                if p_n in self.all_parameters_compute_properties_names:
                    self.parameters_compute_properties[p_n] = p_v

        if pre_trained == True:
            self._is_pre_trained = True
            if pre_trained_pot_filepaths is None:
                raise ValueError('Since you are trying to load a pre-trained model, you must provide the list of paths of the trained-potential files!')
            # turn them into paths
            pre_trained_pot_filepaths = {k: Path(v) for k, v in pre_trained_pot_filepaths.items()}
            self._set_pretrained_model(pre_trained_pot_filepaths=pre_trained_pot_filepaths)
            try:
                self.make_ase_calculator()
            except KeyError:
                print('WARNING: the ASE calculator could not be created because one or more mandatory parameters to compute the properties were missing! Please call  `object.make_ase_calculator(parameters=computing_parameters)`.')
        if save_model == True:
            self.save()

    def _set_pretrained_model(self, pre_trained_pot_filepaths):
        """Configure the model with pre-trained potential files.

        Validates file existence and updates internal paths. This is only used for temporarily loading
        pretrained models before saving them properly.

        Args:
            pre_trained_pot_filepaths (dict): Dictionary of {key: filepath} for trained potential files.
        """
        # check that all the necessary trained_pot files are given
        for k in self.trained_pot_files.keys():
            if k not in pre_trained_pot_filepaths.keys():
                raise ValueError('You are trying to load a pre-trained model, but you did not provide all the required trained-potential files!')
        
        # now let's remove from the list of given files, those who are not required
        extra_keys = [k for k in pre_trained_pot_filepaths if k not in self.trained_pot_files]
        for k in extra_keys:
            del pre_trained_pot_filepaths[k]

        # now we need to be sure that each file is actually there, wherever they are
        for fpath in pre_trained_pot_filepaths.values():
            if not fpath.is_file():
                raise FileNotFoundError(f'The file {fpath.absolute()} does not exist!')
            
        # Trained_pot files are there, so we are confident the model is working.
        # We can set the temporary self.trained_pot_filepaths dictionary. When model will be saved,
        # the files will be copied into the proper trained_pot directory with the proper name
        self.pre_trained_pot_filepaths = pre_trained_pot_filepaths
        
        self.is_trained = True
            

    def set_hyperparameters(self, hyperparameters):
        """Set or update model (training) hyperparameters.

        Args:
            hyperparameters (dict): Dictionary of hyperparameter names and values.
        """
        for k, v in hyperparameters.items():
            if k in self.all_hyperparameters_names:
                self.hyperparameters[k] = v
    
    def set_parameters_compute_properties(self, parameters):
        """Set or update parameters used for computing properties.

        Args:
            parameters (dict): Dictionary of parameter names and values.
        """
        for k, v in parameters.items():
            if k in self.all_parameters_compute_properties_names:
                self.parameters_compute_properties[k] = v


    def _check_mandatory_hyperparameters(self):
        """Check that all mandatory (training) hyperparameters are defined.

        Returns:
            bool or str: True if all mandatory hyperparameters are present;
                otherwise, returns the name of the missing parameter.
        """

        existing_keys = self.hyperparameters.keys() # list of the parameters possessed by self;
                                                    # we need to make sure at least the mandatory ones are there
        for param_name in self.mandatory_hyperparameters_names:
            if param_name not in existing_keys:
                return param_name
        return True
    
    def _check_mandatory_parameters_compute_properties(self, parameters=None):
        """Check that all mandatory parameters for computing properties are defined.

        Args:
            parameters (dict, optional): Parameter dictionary to check.
                If None, uses self.parameters_compute_properties.

        Returns:
            bool or str: True if all mandatory parameters are present;
                otherwise, returns the name of the first missing parameter.
        """
        if parameters is None:
            parameters = self.parameters_compute_properties
        existing_keys = parameters.keys() # list of the parameters possessed by self or passed;
                                            # we need to make sure at least the mandatory ones are there
        for param_name in self.mandatory_parameters_compute_properties_names:
            if param_name not in existing_keys:
                return param_name
        return True

    def get_trained_pot_dir(self):
        return self.root_dir.joinpath(self.trained_pot_dirname).resolve()
    
    def get_training_dir(self):
        return self.root_dir.joinpath(self.training_dirname).resolve()
    
    def get_tmp_dir(self):
        return self.root_dir.joinpath(self.tmp_dirname).resolve()
    
    def get_pot_filepath(self, filename):
        """Get the path of a potential file from its internal name
        
        Returns:
            path (Path): the resolved path of the file
        """
        if not filename in self.trained_pot_files.keys():
            raise RuntimeError(f"The file '{filename}' is not a potential file for the {type(self).class_obj_name} class!")
        if self._is_pre_trained:
            return Path(self.pre_trained_pot_filepaths[filename]).resolve()
        elif self.is_trained:
            return self.get_trained_pot_dir().joinpath(self.trained_pot_files[filename]).resolve()
        else:
            return None
            
            
    def _check_trained_pot_files(self):
        """Verify that all trained potential files exist."""
        for filename in self.trained_pot_files.keys():
            fpath = self.get_pot_filepath(filename)
            # Check: 1. Did we get a path? 2. Does that path exist on disk?
            if fpath is None or not fpath.is_file():
                return False
        return True

          
    def _is_regular(self, level=''):
        """Check whether the model has valid hyperparameters and/or training set.

        Args:
            level (str): Check either full regularity (''), only dataset ('d'), or only hyperparameters ('h').

        Returns:
            bool: True if regular at specified level.
        """
        if level == '':
            if self.regular['d'] == True and self.regular['h'] == True:
                return True
            else:
                return False
        elif level == 'd':
            if self.regular['d'] == True:
                return True
            else:
                return False
        elif level == 'h':
            if self.regular['h'] == True:
                return True
            else:
                return False
        else:
            raise ValueError("Level must be one of: '' (empty string), 'd', or 'h'!")
            
    @abstractmethod
    def _train(self):
        """Abstract method to implement training of the model.

        Returns:
            bool: True if training was successful, False otherwise.
        """
        pass
    
    def _serialize_value(self, val):
        """Recursively serialize non-primitive Python objects for JSON export.

        Args:
            val (any): Value to serialize.

        Returns:
            any: JSON-compatible representation.
        """
        if isinstance(val, Path):
            return {'__type__': 'Path', 'data': str(val.absolute())}
        elif isinstance(val, np.ndarray):
            return {"__type__": "ndarray", "data": val.tolist()}
        elif isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [self._serialize_value(v) for v in val]
        else:
            return val
    
    @classmethod
    def _deserialize_value(cls, val):
        """Recursively reconstruct serialized Python objects from JSON data.

        Args:
            val (any): JSON-loaded value.

        Returns:
            any: Deserialized Python object.
        """
        if isinstance(val, dict) and '__type__' in val.keys():
            if val['__type__'] == 'ndarray':
                return np.array(val['data'])
            elif val['__type__'] == 'Path':
                return Path(val['data'])
            else:
                return val['data'] # this returns the value as it is, regardless of the type 
        elif isinstance(val, dict): # if it is a dictionary that was always meant to be a dictionary
            return {k:cls._deserialize_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [cls._deserialize_value(v) for v in val]
        else:
            return val
        
    def _to_dict(self):
        """Convert the model instance to a dictionary for JSON serialization.

        Excludes attributes not intended for persistence (e.g., train_set).

        Returns:
            dict: Serialized dictionary representation.
        """
        exclude_keys = ['train_set', 'ase_calculator']
        return {k: self._serialize_value(v) for k, v in self.__dict__.items() if k not in exclude_keys}
    
    def save(self, save_train_set_if_pretrained=False, train_set_saving_loc='./'):
        """Serialize and save the model as a JSON file.
        
        Args:
            save_train_set_if_pretrained (bool, optional): True if the training set must be saved; only if self is an unsaved pre-trained model.
            train_set_saving_loc (str or Path, optional): directory where the training set will be saved as 'Training_set.traj'
                if `save_train_set_if_pretrained ` is True.

        """
        if self._is_pre_trained:
            self._store_pretrained_model(lock=True, 
                                     save_train_set=save_train_set_if_pretrained, 
                                     train_set_saving_loc=train_set_saving_loc)
        self_dict = self._to_dict()
        self_dict['class'] = type(self).__name__
        with open(self.root_dir.joinpath(f'{self.name}.json'), 'w') as f:
            json.dump(self_dict, f, indent=2)
            
    def save_copy(self, new_root_dir, new_name=None, extra_dict=None, save_train_set_if_pretrained=False, train_set_saving_loc='./'):
        """Save a copy of MlipModel in a new root directory, including trained potentials."""
        new_root = Path(new_root_dir).resolve()
        new_root.mkdir(exist_ok=True, parents=True)
        
        # 1. PREPARE THE DISK: Copy files FIRST so _from_dict doesn't fail its health checks
        if self.is_trained and not self._is_pre_trained:
            src_dir = self.get_trained_pot_dir()
            dst_dir = new_root.joinpath(self.trained_pot_dirname)
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                
        # 2. EXTRACT STATE: _to_dict() naturally filters out the ase_calculator!
        state_dict = self._to_dict()
        state_dict['class'] = type(self).__name__ # Inject class name for the loader
        
        # 3. RECONSTRUCT: Build the clone pointing to the new directory
        # We pass ignore_metadata=True just to ensure smooth copying, though the files match.
        new_obj = self._from_dict(data=state_dict, 
                                  new_root_dir=new_root, 
                                  train_set=self.train_set, 
                                  ignore_metadata=True)
        if new_name is not None:
            new_obj.name = new_name
        if extra_dict is not None:
            for k, v in extra_dict.items():
                if k in dir(new_obj): # dir(object) gives the name of object's attributes
                    raise KeyError(f'You cannot save a copy with the extra attribute `{k}`, because it is already an attribute of the MlipModel object!')
                setattr(new_obj,k,v)
        
        # 4. FINALIZE: Save the JSON and trigger _store_pretrained_model if necessary
        new_obj.save(save_train_set_if_pretrained=save_train_set_if_pretrained, 
                     train_set_saving_loc=train_set_saving_loc)
                     
        # 5. RE-IGNITE: Spin up a fresh calculator for the new object
        if new_obj.is_trained:
            try:
                new_obj.make_ase_calculator()
            except KeyError:
                print('WARNING: the ASE calculator could not be created because one or more mandatory parameters to compute the properties were missing! Please call  `object.make_ase_calculator(parameters=computing_parameters)`.')
            
        return new_obj

    @classmethod
    def load_model(cls, path, train_set=None, ignore_metadata=False):
        """Load a model from a saved JSON file.

        Args:
            path (str or Path): Path to the saved model JSON.
            train_set (list of Atoms, optional): Training set to load with the model.
            ignore_metadata (bool): Whether to ignore metadata validation. If no metadata is there, set this to False.

        Returns:
            MlipModel: Loaded model object.
        """
        path = Path(path)
        new_obj = cls._from_json(path, train_set, ignore_metadata=ignore_metadata)
        if new_obj.is_trained is True:
            try:
                new_obj.make_ase_calculator()
            except KeyError:
                print('WARNING: the ASE calculator could not be created because one or more mandatory parameters to compute the properties were missing! Please call  `object.make_ase_calculator(parameters=computing_parameters)`.')
        return new_obj
    
    @classmethod
    def _from_json(cls, json_path, train_set=None, ignore_metadata=False):
        """Low-level JSON loader for model deserialization.
        
        Args:
            json_path (str or Path): Path to the JSON file.
            train_set (list of Atoms, optional): External training set to attach (used if metadata matches).
            ignore_metadata (bool): If True, skip metadata validation.
        Returns:
            MlipModel: A reconstructed model object"""
        
        with open(json_path, 'r') as fl:
            data = json.load(fl)
        return cls._from_dict(data=data, 
                              new_root_dir=json_path.parent, 
                              train_set=train_set, 
                              ignore_metadata=ignore_metadata)
        
    @classmethod    
    def _from_dict(cls, data, new_root_dir, train_set=None, ignore_metadata=False):
        """Low-level JSON loader for model deserialization.

        Reads model metadata from a file, reconstructs the model, and validates training consistency.

        Args:
            data (dict): serialized MlipModel.
            new_root_dir (str or Path): root dir of the new object.
            train_set (list of Atoms, optional): External training set to attach (used if metadata matches).
            ignore_metadata (bool): If True, skip metadata validation.

        Returns:
            MlipModel: A reconstructed model object.
        """
        # I just need to change the variable name
        new_train_set = train_set; del train_set
        
        # create a new MlipModel object

        if 'class' not in data.keys():
            pass
        else:
            class_name = data['class']
            del data['class']

            if class_name not in cls._subclass_registry:
               raise ValueError(f"Unknown subclass {class_name} in saved file.")

            real_class = cls._get_class_from_string(cls._subclass_registry[class_name])
            

            # initialize an object with fake root dir, fake name, fake hyperparameters and no dataset
            new_obj = real_class(root_dir='./', name=None, hyperparameters=None, train_set=None)

            # let's give it all his attributes
            for k, v in data.items():
                setattr(new_obj, k, cls._deserialize_value(v))

            # let's set the proper root_dir
            new_obj.root_dir = Path(new_root_dir).resolve()

            # now we need to check if the model is trained. If it is, we need to make sure all the outfiles are there in ./trained_pot/
            if new_obj.is_trained == False:
                new_obj.hash_trained_pot = None
                new_obj.hash_train_set = None
            else:
                if not new_obj._check_trained_pot_files():
                    new_obj.hash_trained_pot = None
                    new_obj.is_trained = False
                    print('The model you are trying to load is trained, but the training-related files are not there. It will be set as untrained.')
                else:
                    trained_pot_filepaths = [new_obj.get_pot_filepath(filename) for filename in new_obj.trained_pot_files.keys()]
                    hash_new_trained_pot = new_obj._make_hash_for_trained_pot(trained_pot_filepaths)
                    if hash_new_trained_pot != new_obj.hash_trained_pot:
                        new_obj.hyperparameters = {}
                        new_obj._make_irregular('h')
                        new_obj.hash_trained_pot = hash_new_trained_pot
                    if ignore_metadata == True:
                        if new_train_set is not None:
                            new_obj.train_set = new_train_set
                            new_obj.hash_train_set = new_obj._make_hash_for_train_set(new_train_set)
                            new_obj._make_metadata()
                        else:
                            # we cannot make the metadata file, because there is no train_set; but let's remove the preexisting one
                            # which may cause abiguities
                            metadata_path = new_obj.get_trained_pot_dir().joinpath('training.metadata') 
                            if metadata_path.is_file():  
                                metadata_path.unlink()
                    else: # check metadata
                        metadata_path = new_obj.get_trained_pot_dir().joinpath('training.metadata') 
                        if metadata_path.is_file():
                            hashes_from_metadata = new_obj._get_hashes_from_metadata()
                            if hashes_from_metadata is not None:
                                if hashes_from_metadata['train_set'] == "None":
                                    hashes_from_metadata['train_set'] = None
                                if hash_new_trained_pot == hashes_from_metadata['trained_pot']:
                                    if new_train_set is not None:
                                        hash_new_train_set = new_obj._make_hash_for_train_set(new_train_set)
                                        if hashes_from_metadata['train_set'] is None:
                                            raise ValueError('This model has no recorded training set. Cannot verify compatibility! Use `ignore_metadata=True` to force load.')
                                        elif hash_new_train_set != hashes_from_metadata['train_set']:
                                            raise ValueError('The train set you provided is incompatible to the metadata file! Use `ignore_metadata=True` to force load.')
                                        else:
                                            new_obj.train_set = new_train_set
                                            new_obj.hash_train_set = hash_new_train_set
                                    else:
                                        if new_obj.hash_train_set is not None:
                                            if new_obj.hash_train_set != hashes_from_metadata['train_set']:
                                                new_obj.hash_train_set = hashes_from_metadata['train_set']
                                                # no need to update the metadata, as we adapted the new_obj.hash_train_set to it
                                            else:
                                                pass # end; no need to update the metadata, as the hashes are already identical
                                        else:
                                            new_obj.hash_train_set = hashes_from_metadata['train_set']
                                            # no need to update the metadata, as we adapted the new_obj.hash_train_set to it
                                else:
                                    raise ValueError('The trained potential was found, but it is incompatible with the metadata file! If you wish to ignore this, load the model with `ignore_metadata` = True.')
                            else: # metadata is a file, but doesn't contain the metadata
                                raise ValueError('The metadata file does not contain metadata info!')
                        else: # metadata is not even a file
                            raise FileNotFoundError(f'No metadata file was found inside {metadata_path.parent.absolute()}!')
            return new_obj


    def _make_hash_for_train_set(self, list_of_structures):
        """Generate a hash that uniquely identifies the given training dataset.

        Args:
            list_of_structures (Atoms or list of Atoms): Dataset to hash.

        Returns:
            str: Hash string for the dataset.
        """
        if isinstance(list_of_structures, Atoms):
            list_of_structures = [list_of_structures]
        return atoms_hash(list_of_structures)
        
    def _make_hash_for_trained_pot(self, list_of_filepaths):
        """Generate a hash for all trained potential files.

        Args:
            list_of_filepaths (list of Path): Paths of trained potential files.

        Returns:
            str: Hash string for the potential files.
        """
        return hash_multiple_files(sorted(list_of_filepaths))
    
    def _make_metadata(self, path=None, hash_trained_pot=None, hash_train_set=None):
        """Write a metadata file containing hashes for the trained potential and training dataset.

        Args:
            path (Path, optional): Output path for the metadata file.
                Defaults to '<trained_pot_dir>/training.metadata'.
            hash_trained_pot (str, optional): Hash of trained potential files.
            hash_train_set (str, optional): Hash of training dataset.
        """

        if path is None:
            path = self.get_trained_pot_dir().joinpath('training.metadata')
        if hash_trained_pot is None:
            hash_trained_pot = self.hash_trained_pot
        if hash_train_set is None:
            hash_train_set = self.hash_train_set
        # store hashes in metadata file
        text = f'Trained_pot_hash:\t{hash_trained_pot}\n'
        text += f'Training_set_hash:\t{hash_train_set}'
        with open(Path(path), 'w') as fl:
            fl.write(text)

    def _get_hashes_from_metadata(self):
        """Read hash values for the trained potential and dataset from metadata.

        Returns:
            dict or None: Dictionary containing 'train_set' and 'trained_pot' hashes,
                or None if the metadata file is missing or invalid.
        """
        path = self.get_trained_pot_dir().joinpath('training.metadata') # cannot use self.metadata_path, because sometimes this function is called before setting it
        with open(path, 'r') as fl:
            #lines = [x.split() for x in fl.readlines()]
            lines = [x.split() for x in fl if x.strip()]
        if lines[0][0] == 'Trained_pot_hash:' and lines[1][0] == 'Training_set_hash:':
            return {'train_set': lines[1][1],'trained_pot':lines[0][1]}
        else:
            return None
        
    def _delete_metadata(self):
        """Delete the training metadata file if it exists."""
        path = self.get_trained_pot_dir().joinpath('training.metadata') # cannot use self.metadata_path, because sometimes this function is called before setting it
        path.unlink()

    def _store_trained_model(self, lock=True, save_train_set=False, train_set_saving_loc="./"):
        """Save trained model files, generate hashes, and optionally save the training set.

        Args:
            lock (bool): If True, store hashes in the model instance.
            save_train_set (bool): Whether to save the training dataset.
            train_set_saving_loc (str or Path): Directory where the training set is saved.
        """
        # if lock=True, a hash number is generated for all the output files and for the dataset and saved as attribute of self. The hashes will be also
        # saved in a metadata file 
        # Beware, without locking, a trained potential can only be used in the current session. If it is saved and reopened, the hashes absence
        # will cause it to be set "untrained". This way we only save the trained pot. files and the train set, for them to be used by the user somehow,
        # but not by the MlipModel object itself! For that, it needs to lock them with hashes. The metadata with the hashes will be saved in the
        # trained pot. dir anyways, so even if the MlipModel object is not locked to the trainset and trained_pot, the last two will be connected to each other by
        # the metadata file. 
        
        # check about the trained potential directory
        if self.get_trained_pot_dir().is_dir():
            shutil.rmtree(self.get_trained_pot_dir())
        self.get_trained_pot_dir().mkdir(parents=True)

        # save trained potential files 
        for v in self.trained_pot_files.values():
            src = self.get_training_dir().joinpath(v) 
            dst = self.get_trained_pot_dir().joinpath(v)
            shutil.copy(src, dst) # copy it in the training dir; after the training it will be copied automatically inside trained_pot_dir 
            # we assumed that when the training is done, the trained_pot files are inside the training_dir and are already named according
            # to self.trained_pot_files.

        # generate hashes
        trained_pot_filepaths = [self.get_pot_filepath(filename) for filename in self.trained_pot_files.keys()]
        hash_trained_pot = self._make_hash_for_trained_pot(trained_pot_filepaths)
        hash_train_set = self._make_hash_for_train_set(self.train_set)

        self._make_metadata(self.get_trained_pot_dir().joinpath('training.metadata'), hash_trained_pot, hash_train_set)

        # store hashes in self
        if lock == True:      
            self.hash_trained_pot = hash_trained_pot
            self.hash_train_set = hash_train_set
        
        # save the training_set
        if save_train_set == True:
            write(Path(train_set_saving_loc).joinpath('Training_set.traj'), self.train_set)

    
    def _store_pretrained_model(self, lock=True, save_train_set=False, train_set_saving_loc='./'):
        """Save pretrained model files, generate hashes, and optionally save the training set.

        Args:
            lock (bool): If True, store hashes in the model instance.
            save_train_set (bool): Whether to save the training dataset.
            train_set_saving_loc (str or Path): Directory where the training set is saved.
        """
        # We need to save a pretrained model. 

        # First let's create the root directory
        self.root_dir.mkdir(exist_ok=True, parents=True)

        # check about the trained potential directory
        if self.get_trained_pot_dir().is_dir():
            shutil.rmtree(self.get_trained_pot_dir())
        self.get_trained_pot_dir().mkdir(parents=True)

        # save trained potential files with proper name in proper directory
        for key in self.trained_pot_files.keys(): 
            value = self.get_pot_filepath(key)
            src = Path(value)
            proper_filename = self.trained_pot_files[key]
            dst = self.get_trained_pot_dir().joinpath(proper_filename)
            shutil.copy(src, dst) # copy the trained pot files in the trained pot dir
        
        self._is_pre_trained = False
        self.is_trained = True
        
        # 1. Generate hashes
        trained_pot_filepaths = [self.get_pot_filepath(filename) for filename in self.trained_pot_files.keys()]
        hash_trained_pot = self._make_hash_for_trained_pot(trained_pot_filepaths)
        
        # If no train_set, we explicitly use None
        hash_train_set = self._make_hash_for_train_set(self.train_set) if self.train_set is not None else None

        # 2. Always write the metadata file (the "Receipt")
        # This creates the file even if hash_train_set is None
        self._make_metadata(path=self.get_trained_pot_dir().joinpath('training.metadata'), 
                            hash_trained_pot=hash_trained_pot, 
                            hash_train_set=hash_train_set)

        # 3. Store hashes in self (The "Lock")
        if lock == True:      
            self.hash_trained_pot = hash_trained_pot
            self.hash_train_set = hash_train_set # This will correctly be None if no train_set
            
        if hasattr(self, 'pre_trained_pot_filepaths'):
            del self.pre_trained_pot_filepaths
            

    def train_model(self, train_set=None, training_hyperparameters=None, save_train_set=False, train_set_saving_loc="./"):
        """Train the model using the provided or stored dataset and hyperparameters.

        Args:
            train_set (list of Atoms, optional): Dataset to train on.
            training_hyperparameters (dict, optional): Training hyperparameters.
            bin_pref (str): Binary prefix for training command.
            save_train_set (bool): Whether to save the training set.
            train_set_saving_loc (str or Path): Path where the training set will be saved as 'Training_set.traj' if `save_train_set` is True.

        Returns:
            bool: True if training succeeded, False otherwise.
        """
        # Three possible models can be there:
        # 1. (all) regular
        # 2. trained but irregular (both partially or totally)
        # 3. untrained (with or without dataset and hyperpars)
        # In any case, if dataset or hyperpars are passed, a backup must be done and restored/deleted if the training is
        # failed/successful. If the training is failed, we must restore the model to it's status before entering in the 
        # training function.
        if self.is_trainable == False:
            raise ValueError('This model is not trainable!')
        # save the status
        train_set_bck = self.train_set
        hyperparameters_bck = self.hyperparameters
        
        if self.get_training_dir().is_dir():
            shutil.rmtree(self.get_training_dir().absolute())
        self.get_training_dir().mkdir(parents=True)
        
        if train_set is None:
            if self.train_set is None:
                raise ValueError('To train the potential, either the model object must have a training set, or the parameter `train_set` must be passed to this method.')
            else:
                txt1 = ''
                txt2 = ''
        else:
            if len(train_set) < self.min_train_set_size:
                raise ValueError(f'The training set must have at least {self.min_train_set_size} configurations; you provided only {len(train_set)}.')
            if self.train_set is not None:
                txt1 = '\nThe preexisting trainset has been overwritten with the new dataset provided.'
                txt2 = '\nThe preexisting trainset was preserved (not replaced by the new dataset provided).'
            else:
                txt1 = ''
                txt2 = ''
            self.train_set = train_set

        if training_hyperparameters is None:
            if self.hyperparameters is None:
                raise ValueError('To train the potential, either the model object must have a hyperparameters, or the parameter `training_hyperparameters` must be passed to this method.')    
            else:
                txt3 = ''
                txt4 = ''
        else:
            if self.hyperparameters is not None:
                txt3 = '\nThe preexisting hyperparameters have been overwritten with the new ones provided.'
                txt4 = '\nThe preexisting hyperparameters were preserved (not replaced by the new ones provided).'
            else:
                txt3 = ''
                txt4 = ''
            self.hyperparameters = training_hyperparameters
       
       # we need to check if all the mandatory hyperparameters are there (we already checked that no extra, unkown parameters were given)
        res = self._check_mandatory_hyperparameters()
        if res is not True: # then `res` is the name of the missing mandatory hyperparameter
            raise KeyError(f'The {type(self).class_obj_name} object has no hyperparameter called `{res}` and you didn\'t provide any!')
 
        # let's create the trained_pot directory
        self.get_trained_pot_dir().mkdir(parents=True, exist_ok=True)
        # run the training
        try:
            success = self._train() # the implementation needs to have (only) this argument: bin_pref!!!
        except:
            success = False
            # we need to restore the old things
            self.train_set = train_set_bck
            self.hyperparameters = hyperparameters_bck
        if success == False:
            print('The training was not successful.' + txt2 + txt4)
            print('Please check the log files to understand why.')
            return success
        
        # from here only if sucess is True
            # if new trainset or hyperpars were given, nore they are already set as self's attributes; we only need to regularize and store everything
        
        print('Training done successfully' + txt1 + txt3)
        self.is_trained = True
        if self._is_pre_trained == True:
            self._is_pre_trained = False
        del train_set_bck
        del hyperparameters_bck
        self._make_regular()
        self._store_trained_model(save_train_set=save_train_set, train_set_saving_loc=train_set_saving_loc) # store trained info - ! must be before the call to save !
        self.save() # save the object as dictionary
        if hasattr(self, 'ase_calculator'): # this wouldn't be saved anyways
            del self.ase_calculator # it's better to remove it, because of possible cache issues 
        try:
            self.make_ase_calculator()
        except KeyError:
            print('WARNING: the ASE calculator could not be created because one or more mandatory parameters to compute the properties were missing! Please call  `object.make_ase_calculator(parameters=computing_parameters)`.')
        return success
            

    def _make_regular(self, level=''):
        """Mark the model as regular (i.e., consistent and up to date).

        Args:
            level (str): '' for all, 'd' for dataset, 'h' for hyperparameters.
        """
        if level == '':
            self.regular['d'] = True
            self.regular['h'] = True
        elif level == 'd':
            self.regular['d'] = True
        elif level == 'h':
            self.regular['h'] = True
    
    def _make_irregular(self, level=''):
        """Mark the model as irregular (i.e., outdated or inconsistent).

        Args:
            level (str): '' for all, 'd' for dataset, 'h' for hyperparameters.
        """
        if level == '':
            self.regular['d'] = False
            self.regular['h'] = False
        elif level == 'd':
            self.regular['d'] = False
        elif level == 'h':
            self.regular['h'] = False
    

    def _general_compute_properties(self, atoms, wdir, parameters=None, mute=True, function=None, update_parameters=False):
        """General function to compute properties. A function must be passed. This function is either `self._compute_properties_list` or `self._compute_properties_single`.
           This function is not meant for the user!
        Args:
            atoms (Atoms): Structure to evaluate.
            wdir (str or Path): Working directory for temporary files.
            parameters (dict, optional): Parameters for computation.

        Returns:
            Atoms: Structure with computed properties (list or Atoms based on `function`).
        """
        if not self.is_trained:
            raise ValueError('The calculation cannot be done because the model is not trained!')
        if function is None:
            function = self._compute_properties
        wdir = Path(wdir)
        if parameters is None:
            # we need to check if all the mandatory perparameters are there
            res = self._check_mandatory_parameters_compute_properties(parameters = self.parameters_compute_properties)
            to_pass = self.parameters_compute_properties.copy()
        else:
            res = self._check_mandatory_parameters_compute_properties(parameters = parameters)
            to_pass = parameters
        if res is not True: # then `res` is the name of the missing mandatory parameter
            raise KeyError(f'The {type(self).class_obj_name} object has no parameter called `{res}` for computing properties, and you didn\'t provide any!')

        # !!! If parameters are given, some extra unknown parameters could have been passed, but that's not important, as in self._compute_properties only the proper ones are used
        # However, if parameters were given to the model when initialized, any extra one was removed.
        to_pass['mute'] = mute
        atoms_calc = function(atoms=atoms, wdir=wdir, parameters=to_pass)
        return atoms_calc

    def compute_properties(self, atoms, wdir, parameters=None, mute=True):
        '''
        This is supposed to be the most used by the user; hence it accepts both lists and Atoms objects.  

        Args:
        atoms(ase.atoms.Atoms or List): ase.atoms.Atoms object or list of Atoms objects 
        
        Return
        atoms_calcase.atoms.Atoms or List): configuration(s) with computed properties (Atoms or List, based on `atoms`)
        '''
        if isinstance (atoms, Atoms):
            atoms_calc = self.compute_properties_single_structure(atoms, wdir, parameters=parameters, mute=mute)
        else:
            atoms_calc = self._general_compute_properties([x.copy() for x in atoms], wdir, parameters=parameters, mute=mute, function=self._compute_properties_list)
        return atoms_calc

    def compute_properties_single_structure(self, atoms, wdir, parameters=None, mute=True):
        '''
        This is CAN be used by the user, if he is aware he only needs one structure. However, `compute_properties` contains this one as well. If not for efficiency issues, this function can be ignored.

        Args:
        atoms(ase.atoms.Atoms): Atoms object

        Return
        atoms_calc(ase.atoms.Atoms) = Atoms object with computed properties (SinglePointCalculator)
        '''
        atoms_calc = self._general_compute_properties(atoms.copy(), wdir, parameters=parameters, mute=mute, function=self._compute_properties_single)
        return atoms_calc
 

    @abstractmethod
    def _compute_properties_single(self, atoms, wdir, **kwargs):
        pass    

    @abstractmethod
    def _compute_properties_list(self, atoms, wdir, **kwargs):
        pass

    
    def evaluate_on_dataset(self, dataset, wdir, parameters=None, save_results=True):
        """Evaluate the trained model on a dataset and compute error metrics.

        Args:
            dataset (Atoms or list of Atoms): Dataset with reference properties.
            wdir (str or Path): Working directory to store results.
            parameters (dict, optional): Parameters for computing properties.
            save_results (bool): Whether to save evaluation files.

        Returns:
            tuple: (evaluated structures, error metrics dictionary) (ASE units)
                for a given element of the error dictionary, a list [rmse, mae, R2] is given.
        """

        if not self.is_trained:
            raise ValueError('The model is not trained!')
        
        wdir = Path(wdir)
        wdir.mkdir(parents=True, exist_ok=True)

        if isinstance(dataset, Atoms):
            dataset = [dataset]

        res_structs = self.compute_properties(atoms=dataset, wdir=wdir, parameters=parameters)
        errs = make_comparison(structures1 = dataset,
                               structures2 = res_structs,
                               props=self.computable_properties_names,
                               make_file=save_results,
                               dir=wdir,
                               outfile_pref='')
        
        # errs is a dictionary whose key/vlaue elements are property/errors, where property can be 
        # {'energy', 'forces', 'stress'}, and errors is a list [rmse, mae, R2]
 
            
        return res_structs, errs

    def make_ase_calculator(self, parameters=None):
        if parameters is not None:
            res = self._check_mandatory_parameters_compute_properties(parameters)
        else:
            res = self._check_mandatory_parameters_compute_properties()
        
        if res is not True: # then `res` is the name of the missing mandatory parameter
            raise KeyError(f'The {type(self).class_obj_name} object has no parameter called `{res}` for computing properties, and you didn\'t provide any!')
        if parameters is not None:
            self.parameters_compute_properties = parameters
        self.ase_calculator = MlipkitCalculator(mlipkit_model=self)
        return self.ase_calculator

    
    # def _set_trainset(self, dataset):
    #     if (not self.is_trained and self.train_set is not None) or self._is_regular('d'):
    #         self.trainset_bck = self.train_set.copy()
    #     self.train_set = dataset
    #     self._make_irregular('d')
    
    # def _set_hyperparameters(self, hyperparameters):
    #     if (not self.is_trained and self.hyperparameters is not None) or self._is_regular('h'):
    #         self.hyperparameters_bck = self.hyperparameters.copy()
    #     self.hyperparameters = hyperparameters
    #     self._make_irregular('h')
    
    # def _delete_backups(self, level=''):
    #     if level == '':
    #         self.hyperparameters_bck = None
    #         self.trainset_bck = None
    #     elif level == 'd':
    #         self.trainset_bck = None
    #     elif level == 'h':
    #         self.hyperparameters_bck = None


    # def _unset_trainset(self):
    #     """Remove the active training dataset reference without deleting its backup."""
    #     self.train_set = None
    #     self._is_regular = True
    
    # def _reset_trainset_bck(self):
    #     '''If there is self.trainset_bck, it is assigned to self.train_set and then removed'''
    #     print('Reset of the previous train set backup')
    #     if self.trainset_bck is not None:
    #         self.train_set = self.trainset_bck
    #         self.trainset_bck = None
    #     else:
    #         print('No preexisting train set backup; aborted!')

    # def _reset_hyperparameters_bck(self):
    #     '''If there is self.hyperparameters_bck, it is assigned to self.hyperparameters and then removed'''
    #     print('Reset of the previous hyperparameters backup')
    #     if self.hyperparameters_bck is not None:
    #         self.hyperparameters = self.hyperparameters_bck.copy()
    #         self.hyperparameters_bck = None
    #     else:
    #         print('No preexisting hyperparameters backup; aborted!')

