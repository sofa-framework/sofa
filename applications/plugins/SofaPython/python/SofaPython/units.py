## Converting SI units to/from specific units
# Every units are given as a ratio to the SI units (m/kg/s)
#
# For each conversion a local unit can be given
# OR
# variables local_time/local_length/local_mass are used by default
# (these variables can be set by the user to define local units)
#
# Note that Poisson ratio and strain have no units

# TODO have a look to units for stress and elasticity parameters in 2D and 1D



import math


### some CONSTANTS are defined as user utilities
# ratio to the SI units

# LENGTH
length_hm  = 1e2
length_dam = 1e1
length_m   = 1.0
length_dm  = 1e-1
length_cm  = 1e-2
length_mm  = 1e-3
imperial_length_ft = 0.3048                  # foot
imperial_length_in = imperial_length_ft/12.0 # inch
imperial_length_yd = imperial_length_ft*3.0  # yard
imperial_length_mi = imperial_length_ft*5.28 # mile

# MASS
mass_g   = 1e-3
mass_kg  = 1.0
mass_ton = 1e3
imperial_mass_lb  = 0.45359237               # pound
imperial_mass_oz  = imperial_mass_lb/16.0    # ounce
imperial_mass_ton = imperial_mass_lb*2240.0 # imperial ton

# TIME
time_h   = 3600.0
time_min = 60.0
time_s   = 1.0
time_ms  = 1e-3




### LOCAL UNITS
# user variables to set to describe a set of local units

local_time = time_s
local_length = length_m
local_mass = mass_kg




### conversion methods

# BASE

# s
def time_from_SI( t, time_unit=None ):
    time_unit = time_unit or local_time
    return t / time_unit

def time_to_SI( t, time_unit=None ):
    time_unit = time_unit or local_time
    return t * time_unit

# m
def length_from_SI( l, length_unit=None ):
    length_unit = length_unit or local_length
    return l / length_unit

def length_to_SI( l, length_unit=None ):
    length_unit = length_unit or local_length
    return l * length_unit

# kg
def mass_from_SI( m, mass_unit=None ):
    mass_unit = mass_unit or local_mass
    return m / mass_unit

def mass_to_SI( m, mass_unit=None ):
    mass_unit = mass_unit or local_mass
    return m * mass_unit




# DERIVED

# m2
def area_from_SI( a, length_unit=None ):
    length_unit = length_unit or local_length
    return a / length_unit / length_unit

def area_to_SI( a, length_unit=None ):
    length_unit = length_unit or local_length
    return a * length_unit * length_unit

# m3
def volume_from_SI( v, length_unit=None ):
    length_unit = length_unit or local_length
    return v / length_unit / length_unit / length_unit

def volume_to_SI( v, length_unit=None ):
    length_unit = length_unit or local_length
    return v * length_unit * length_unit * length_unit



# m/s
def velocity_from_SI( v, length_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    time_unit = time_unit or local_time
    return v / length_unit * time_unit

def velocity_to_SI( v, length_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    time_unit = time_unit or local_time
    return v * length_unit / time_unit

# m/s2
def acceleration_from_SI( v, length_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    time_unit = time_unit or local_time
    return v / length_unit * time_unit * time_unit

def acceleration_to_SI( v, length_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    time_unit = time_unit or local_time
    return v * length_unit / time_unit / time_unit



# N = kg.m/s2
def force_from_SI( f, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return f / length_unit / mass_unit * time_unit * time_unit

def force_to_SI( f, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return f * length_unit * mass_unit / time_unit / time_unit

# C = N.m = kg.m2/s2
def torque_from_SI( c, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return c / length_unit / length_unit / mass_unit * time_unit * time_unit

def torque_to_SI( c, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return c * length_unit * length_unit * mass_unit / time_unit / time_unit

# Pa = N/m2 = kg/(m.s2)
def pressure_from_SI( p, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return p / mass_unit * length_unit * time_unit * time_unit

def pressure_to_SI( p, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return p * mass_unit / length_unit / time_unit / time_unit


# J = N.m = kg.m2/s2
def energy_from_SI( e, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return e / length_unit / length_unit / mass_unit * time_unit * time_unit

def energy_to_SI( e, length_unit=None, mass_unit=None, time_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    time_unit = time_unit or local_time
    return e * length_unit * length_unit * mass_unit / time_unit / time_unit

# I = kg.m2
def inertia_from_SI( i, length_unit=None, mass_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    return i / length_unit / length_unit / mass_unit 

def inertia_to_SI( i, length_unit=None, mass_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    return i * length_unit * length_unit * mass_unit 



# MATERIAL

# Pa
def elasticity_from_SI( e, length_unit=None, mass_unit=None, time_unit=None ):
    ## works for Young/bulk/shear modulus and Lame parameters
    return pressure_from_SI( e, length_unit, mass_unit, time_unit )

def elasticity_to_SI( e, length_unit=None, mass_unit=None, time_unit=None ):
    ## works for Young/bulk/shear modulus and Lame parameters
    return pressure_to_SI( e, length_unit, mass_unit, time_unit )

# kg/m3
def density_from_SI( d, length_unit=None, mass_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    return d * length_unit * length_unit * length_unit / mass_unit

def density_to_SI( d, length_unit=None, mass_unit=None ):
    length_unit = length_unit or local_length
    mass_unit = mass_unit or local_mass
    return d / length_unit / length_unit / length_unit * mass_unit

# w/o
def poissonRatio_from_SI( pr ):
    return pr

def poissonRatio_to_SI( pr ):
    return pr


# MECHANICS

# Pa
def stress_from_SI( s, length_unit=None, mass_unit=None, time_unit=None ):
    return pressure_from_SI( s, length_unit, mass_unit, time_unit )

def stress_to_SI( s, length_unit=None, mass_unit=None, time_unit=None ):
    return pressure_to_SI( s, length_unit, mass_unit, time_unit )

# w/o
def strain_from_SI( s ):
    return s

def strain_to_SI( s ):
    return s

# m
def deformation_from_SI( d, length_unit=None ):
    return length_from_SI( d, length_unit )

def deformation_to_SI( d, length_unit=None ):
    return length_to_SI( d, length_unit )
