
from ase.calculators.vasp import Vasp
from ase.atoms import Atoms
import os

# Mock VASP_PP_PATH if needed, though Vasp calculator might just check internal dicts
# We just want to see the default setups
calc = Vasp()
print("Default setups:", calc.input_params.get('setups'))

# To see specific element mapping, we might need to look at ase.calculators.vasp.create_input
from ase.calculators.vasp.create_input import get_vasp_setup

# This is internal ASE logic, might vary by version.
# Let's just try to see what it selects for Y if we were to write an input.
# We need a fake atom
atoms = Atoms('Y')
calc = Vasp(xc='PBE')
calc.initialize(atoms)
print("Selected POTCAR for Y:", calc.sort_pompot(atoms))
