import sys
sys.path.append("../python")

import SofaPython.units

# some physical parameters
# given in SI units
voxelSize=0.05
g=9.81
youngModule=50e5

# setup local units
SofaPython.units.local_length = SofaPython.units.length_mm
SofaPython.units.local_mass = SofaPython.units.mass_g
SofaPython.units.local_time = SofaPython.units.time_ms

# get the proper values to setup the scene
SofaPython.units.length_from_SI(voxelSize)
SofaPython.units.acceleration_from_SI(g)
SofaPython.units.elasticity_from_SI(youngModule)

# run the simulation...
velocity=10 # mm/ms
force = 50 # g.mm/ms2

