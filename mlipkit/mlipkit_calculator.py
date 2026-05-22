import shutil
from ase.calculators.calculator import Calculator


class MlipkitCalculator(Calculator):
    """
    ASE-compatible calculator wrapper for Mlipkit models.

    This class bridges the Mlipkit model logic with the ASE Calculator interface, 
    enabling standard ASE workflows such as geometry optimizations, molecular 
    dynamics, and vibration analysis. It acts as a lightweight proxy that 
    delegates physics calculations to the underlying MLIP model.

    Attributes:
        implemented_properties (list): Standard properties supported by the 
            calculator: ['energy', 'forces', 'stress'].

    Methods:
        calculate(atoms, properties, system_changes): Primary ASE entry point. 
            Runs the MLIP model on a single structure, updates the results 
            dictionary, and cleans up temporary working files.
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, mlipkit_model):
        Calculator.__init__(self)
        self.mlipkit_model = mlipkit_model
        
    def calculate(self,
                  atoms,
                  properties=['energy', 'forces', 'stress'],
                  system_changes = ['positions', 'numbers', 'cell', 'pbc']):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        calculated_atoms =  self.mlipkit_model.compute_properties_single_structure(atoms, self.mlipkit_model.tmp_dir)
        
        energy = calculated_atoms.get_potential_energy()
        forces = calculated_atoms.get_forces()
        stress = calculated_atoms.get_stress()
        if self.mlipkit_model.tmp_dir.is_dir():
            shutil.rmtree(self.mlipkit_model.tmp_dir.resolve())
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
