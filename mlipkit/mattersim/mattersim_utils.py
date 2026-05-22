import torch
import os
import re
import numpy as np
from subprocess import run
from pathlib import Path
from ase.io import read, write






def make_comparison(structures1=None, 
                    structures2=None, 
                    props='all', 
                    make_file=False, 
                    dir='',
                    outfile_pref='', 
                    units=None):
    """
    Generates parity data and computes error metrics between reference and predicted structures.

    This function extracts properties (energy, forces, stress) from two sets of 
    ASE Atoms objects, calculates standard regression metrics (RMSE, MAE, R2), 
    and optionally writes the raw comparison data to disk for plotting.

    Args:
        structures1 (ase.Atoms or list of ase.Atoms): Ground truth (reference) configurations.
        structures2 (ase.Atoms or list of ase.Atoms): Predicted (ML) configurations.
        props (str or list): Properties to compare. Options are 'energy', 'forces', 
            'stress', or 'all'. Defaults to 'all'.
        make_file (bool): If True, writes comparison data to .dat files.
        dir (str or Path): Output directory for .dat files. Required if make_file is True.
        outfile_pref (str): Prefix for generated filenames (e.g., 'MTP-').
        units (dict, optional): Custom units for the header. Defaults to 
            eV/atom for energy, eV/Angstrom for forces, and GPa for stress.

    Returns:
        dict: dictionary whose key/vlaue elements are property/errors, where property can be  {'energy', 'forces', 'stress'}, and errors is a list [rmse, mae, R2].

    Raises:
        AssertionError: If structures are missing or counts do not match.
        ValueError: If 'dir' is missing while make_file is True, or if invalid 
            properties are requested.
    """
    
    assert (structures1 is not None), f"You must give the `structures1 `!"
    if isinstance(structures1, Atoms):
            structures1 = [structures1]
        
    assert (structures2 is not None), f"You must give the `structures2 `!"
    if isinstance(structures2, Atoms):
            structures2 = [structures2]    
    
    if make_file == True:
        if dir is not None:
            dir = Path(dir)
        else:
            raise ValueError('You must give `dir` since `make_file` is True!')
                    
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
    ext1 = [flatten(x) for x in extract_prop_from_ase(structures1)]
    ext1[0] = ext1[0] #* natoms

    ext2 = [flatten(x) for x in extract_prop_from_ase(structures2)]
    ext2[0] = ext2[0] #* natoms

    assert len(ext1) == len(ext2), f"You gave a different number of "\
        + f"true and ML structures!"

    # Compute errors and write data on files
    errs = dict()
    for prop in props:
        i = prop_numbs[prop]
        mae2 = mae(ext1[i], ext2[i])
        rmse2 = rmse(ext1[i], ext2[i])
        R22 = R2(ext1[i], ext2[i])
        errs[prop] = [rmse2, mae2, R22]
        
        if make_file == True:
            filename = dir.joinpath(f'{outfile_pref}{prop.capitalize()}_comparison.dat')
            print(f'printing in {filename.absolute()}')
            text = f'# rmse: {rmse2:.5f} {units[prop]},    mae: {mae2:.5f} {units[prop]}    R2: {R22:.5f}\n'
            text += f'#  True {low_first(prop)}           Predicted {low_first(prop)}\n'
            for x, y in zip(ext1[i], ext2[i]):
                text += f'{x:.20f}  {y:.20f}\n'
            with open(filename.absolute(), 'w') as fl:
                fl.write(text)
    return errs

# def check_cuda_state():
#     """
#     Performs a diagnostic check of the CUDA environment and GPU process state.

#     This function is primarily used to debug issues with CUDA initialization 
#     and multi-process GPU access (e.g., when using torchrun or SLURM). It 
#     reports the initialization status of the torch.cuda module, current 
#     memory allocation, and identifies other PIDs currently utilizing the GPU 
#     via NVML.

#     Returns:
#         None

#     Note:
#         - Memory allocation checks are only performed if CUDA is available.
#         - Process tracking requires the 'pynvml' package to be installed; 
#           otherwise, this section of the diagnostic is skipped.
#         - This function prints information directly to the standard output.
#     """
    
#     print("--- CUDA PRE-INIT CHECK ---")
#     # 1. Check if the torch.cuda module itself has been initialized
#     print(f"Is torch.cuda initialized? {torch.cuda.is_initialized()}")
    
#     # 2. Check current memory allocated in the parent process
#     if torch.cuda.is_available():
#         try:
#             current_mem = torch.cuda.memory_allocated()
#             print(f"Parent process GPU memory allocation: {current_mem} bytes")
#         except Exception as e:
#             print(f"Could not check memory (likely context error): {e}")
    
#     # 3. Check for existing context via NVML (if installed)
#     try:
#         import pynvml
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming GPU 0
#         procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
#         print(f"Processes already using this GPU: {[p.pid for p in procs]}")
#     except:
#         print("NVML not available for process check.")
#     print("---------------------------")


# !!!! finetune_pot assumes only one GPU node is allocated

# !!!! When running on cpus, it assumes only 1 task and N cpus-per-task (i.e. OMP not MPI)
# you always need to specify the variables --ntasks=1 and --cpus-per-task=N. This will be read by the python code below!
def finetune_pot(wdir,
                finetune_script_path=None,
                run_name = 'finetune',
                train_set = None,
                test_set = None,
                load_model_path = 'mattersim-v1.0.0-1m',
                save_checkpoint = True,
                saving_path = None,
                ckpt_interval = 10,
                device = 'cuda',
                cutoff = 5,
                threebody_cutoff = 4,
                epochs = 1000,
                batch_size = 16,
                learning_rate = 2e-4,
                step_size = 10,
                include_forces = True,
                include_stresses = True,
                force_loss_ratio = 1,
                stress_loss_ratio = 0.1,
                early_stop_patience = 10,
                seed = 42,
                re_normalize = False,
                scale_key = 'per_species_forces_rms',
                shift_key = 'per_species_energy_mean_linear_reg',
                init_scale = None,
                init_shift = None,
                trainable_scale = False,
                trainable_shift = False,
                wandb = False,
                wandb_api_key = None,
                wandb_project = 'wandb_test',
                delete_sets = True,
                final_evaluation=False,
                eval_dir=None,
                **kwargs):
    """
    Orchestrates the fine-tuning of a MatterSim model using torchrun.

    This utility prepares the environment, handles dataset serialization to 
    temporary files, and executes the fine-tuning script as a subprocess. It 
    automatically manages Distributed Data Parallel (DDP) scaling based on 
    available GPU resources (SLURM or local environment).

    Args:
        wdir (str or Path): Working directory for logs and temporary files.
        finetune_script_path (str or Path): Path to the MatterSim fine-tuning script.
        run_name (str): Identifier for the training run and output subdirectories.
        train_set (list of ase.Atoms): Configurations used for model training.
        test_set (list of ase.Atoms): Configurations used for validation.
        load_model_path (str or Path): Path to starting weights or model identifier.
        save_checkpoint (bool): If True, enables saving model checkpoints.
        saving_path (str or Path, optional): Path where results are saved. 
            Defaults to wdir/training_results.
        ckpt_interval (int): Frequency (in epochs) to save checkpoints.
        device (str): Computation device ('cuda' or 'cpu').
        cutoff (float): Cutoff distance for the atomic neighbor list.
        threebody_cutoff (float): Cutoff distance for three-body interactions.
        epochs (int): Total number of fine-tuning iterations.
        batch_size (int): Number of structures per training batch.
        learning_rate (float): Initial learning rate for the optimizer.
        step_size (int): Epoch interval for learning rate decay.
        include_forces (bool): Whether to include forces in the loss function.
        include_stresses (bool): Whether to include stress in the loss function.
        force_loss_ratio (float): Weight of the force error in total loss.
        stress_loss_ratio (float): Weight of the stress error in total loss.
        early_stop_patience (int): Epochs to wait for improvement before stopping.
        seed (int): Random seed for reproducibility.
        re_normalize (bool): Whether to re-calculate data normalization stats.
        scale_key (str): Key for force scaling stats in the state dict.
        shift_key (str): Key for energy shift stats in the state dict.
        init_scale (float, optional): Initial manual scaling factor.
        init_shift (float, optional): Initial manual energy shift.
        trainable_scale (bool): Whether scaling factors are optimized.
        trainable_shift (bool): Whether energy shifts are optimized.
        wandb (bool): Enable Weights & Biases logging.
        wandb_api_key (str, optional): API key for W&B authentication.
        wandb_project (str): W&B project name.
        delete_sets (bool): If True, temporary training set and test sets will be deleted 
        final_evaluation (bool): If True, evaluates the best model on the test set.
        eval_dir (str or Path, optional): Directory for evaluation results.
        **kwargs: Additional arguments passed to the fine-tuning command line.

    Returns:
        Path: Path to the generated 'best_model.pth'.
        tuple: (best_model_path, computed_structures, errs) if final_evaluation is True.

    Raises:
        ValueError: If load_model_path is None or validation split is missing.
        FileNotFoundError: If the initial model weights are not found.
    """
    params = locals().copy() # !!! this must be the first instruction !!!
    def get_flags(params):
        flags = dict(run_name = '--run_name',
                     train_data_path = '--train_data_path',
                     valid_data_path = '--valid_data_path',
                     load_model_path = '--load_model_path',
                     save_path = '--save_path',
                     save_checkpoint = '--save_checkpoint',
                     ckpt_interval = '--ckpt_interval',
                     device = '--device',
                     cutoff = '--cutoff',
                     threebody_cutoff = '--threebody_cutoff',
                     epochs = '--epochs',
                     batch_size = '--batch_size',
                     learning_rate = '--lr',
                     step_size = '--step_size',
                     include_forces = '--include_forces',
                     include_stresses = '--include_stresses',
                     force_loss_ratio = '--force_loss_ratio',
                     stress_loss_ratio = '--stress_loss_ratio',
                     early_stop_patience = '--early_stop_patience',
                     seed = '--seed',
                     re_normalize = '--re_normalize',
                     scale_key = '--scale_key',
                     shift_key = '--shift_key',
                     init_scale = '--init_scale',
                     init_shift = '--init_shift',
                     trainable_scale = '--trainable_scale',
                     trainable_shift = '--trainable_shift',
                     wandb = '--wandb',
                     wandb_api_key = '--wandb_api_key',
                     wandb_project = '--wandb_project')

        # first let's set the boolean parameters
        boolean_parameters = ['save_checkpoint', 'include_forces', 'include_stresses', 're_normalize', 'trainable_scale', 'trainable_shift', 'wandb']
        cmd = ''
        for par in list(params.keys()):
            if par in boolean_parameters:
                if params[par] == True:
                    cmd = f'{cmd} {flags[par]}'
                else:
                    pass
            elif par in list(flags.keys()): # not boolean but still a valid parameter
                if params[par] is None:
                    pass # we simply don't write anything to let mattersim use the default value
                else:
                    cmd = f'{cmd} {flags[par]} {params[par]}'
        return cmd
    
    if 'env' not in kwargs.keys():
        env = os.environ.copy()
    else:
        env = kwargs['env']
    finetune_script_path = Path(finetune_script_path)
    load_model_path = str(Path(load_model_path).resolve())
    wdir = Path(wdir)
    wdir.mkdir(parents=True, exist_ok=True)
    if saving_path is None:
        saving_path = wdir.joinpath('training_results').resolve()
    else:
        saving_path = Path(saving_path).resolve()
    saving_path.mkdir(parents=True, exist_ok=True)
    params['save_path'] = saving_path
    
    if final_evaluation == True:
        if eval_dir is None:
            eval_dir = wdir.joinpath('evaluation')
        else:
            eval_dir = Path(eval_dir) 
    
    # we must write train and test set for mattersim to read it, remember to delete them, at the end
    write(wdir.joinpath('Training_set.traj'), train_set)
    write(wdir.joinpath('Test_set.traj'), test_set)

    params['train_data_path'] = wdir.joinpath('Training_set.traj').absolute()
    params['valid_data_path'] = wdir.joinpath('Test_set.traj').absolute()
    
    # before getting the string of flags, let's check some parameters
    if load_model_path is None:
        raise ValueError('You must give the `load_model_path`!')
    elif not Path(load_model_path).is_file():
        raise FileNotFoundError(f'The file {Path(load_model_path)} does not exist!')


    flags = get_flags(params)
    if device == 'cuda':
        vis_devs = env.get('CUDA_VISIBLE_DEVICES')
        # if the visible gpus are restricted by CUDA_VISIBLE_DEVICES, let's use them
        if vis_devs:
            n_proc = len(vis_devs.split(','))
        else: # otherwise we assume all available gpus are to be used, so let's count them from slurm
            raw_val = os.environ.get('SLURM_GPUS_ON_NODE', '1')
            n_proc = re.findall(r'\d+', raw_val)[0]
        device_torchrun_spec = f'--nproc_per_node={n_proc}'
        #device_torchrun_spec = '--nproc_per_node=1'
    else:
        raw_val = os.environ.get('SLURM_CPUS_PER_TASK', '1')
        n_cores = re.findall(r'\d+', raw_val)[0]
        #print('ncores: ', str(n_cores))
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
        os.environ["MKL_NUM_THREADS"] = str(n_cores)
        device_torchrun_spec = '--nproc_per_node=1' # with cpus, we use OMP with just one process and N threads
        
    cmd = f'torchrun {device_torchrun_spec} {finetune_script_path} {flags}'
    log_path = wdir.joinpath('log_train')
    err_path = wdir.joinpath('err_train')
    #print(f"Command for the training: {cmd}")
    variables_to_check = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES', 'NCCL_DEBUG', 'NCCL_IB_DISABLE', 'SLURM_GPUS_ON_NODE', 'SLURM_JOB_ID', 'SLURM_STEP_GPUS']
    #variables_string = ''.join([x + ': ' + str(os.environ.get(x)) + '|\n' for x in variables_to_check])
    #print(f"Environmental variables.\n{variables_string}")
    #check_cuda_state()
    with open(log_path.absolute(), 'w') as log, open(err_path.absolute(), 'w') as err:
        run(cmd.split(), env=env, cwd=wdir.absolute(), stdout=log, stderr=err)
     
    trained_model_path = saving_path.joinpath('best_model.pth')
    
    if final_evaluation == True:
        if not eval_dir.is_dir():
            eval_dir.mkdir(parents=True, exist_ok=True)

        computed_structures = calc_efs()
        
        errs = make_comparison() # probably will need computed_structures
        
    # before returning, let's delete the datasets
    if delete_sets == True:
        wdir.joinpath('Training_set.traj').unlink(missing_ok=True)
        wdir.joinpath('Test_set.traj').unlink(missing_ok=True)
    if final_evaluation == True:
        return trained_model_path, computed_structures, errs
    else:
        return trained_model_path
