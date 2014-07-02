import SofaPython.units as units

from SofaTest.Macro import *

def run():

    ok=True

    # setting local units (as ratio to SI units m/kg/s)
    units.local_length = units.length_cm
    units.local_mass   = units.mass_g
    units.local_time   = units.time_min

    ### BASE

    # lenght
    ok &= EXPECT_FLOAT_EQ( 1000, units.length_from_SI( 1, units.length_mm ), "length from SI local" ) # converting 1 m in mm
    ok &= EXPECT_FLOAT_EQ( 100, units.length_from_SI( 1 ), "length from SI global" ) # converting 1 m in cm
    ok &= EXPECT_FLOAT_EQ( 23.12, units.length_from_SI( units.length_to_SI( 23.12 ) ), "length from_to SI global" ) # converting 23.12 m in cm then in m

    # mass
    ok &= EXPECT_FLOAT_EQ( 1e-3, units.mass_from_SI( 1, units.mass_ton ), "mass from SI local" ) # converting 1 kg in ton
    ok &= EXPECT_FLOAT_EQ( 1000, units.mass_from_SI( 1 ), "mass from SI global" ) # converting 1 kg in g
    ok &= EXPECT_FLOAT_EQ( 23.12, units.mass_from_SI( units.mass_to_SI( 23.12 ) ), "mass from_to SI global" ) # converting 23.12 kg in g then in kg

    # time
    ok &= EXPECT_FLOAT_EQ( 1.0/3600.0, units.time_from_SI( 1, units.time_h ), "time from SI local" ) # converting 1 s in hour
    ok &= EXPECT_FLOAT_EQ( 1.0/60.0, units.time_from_SI( 1 ), "time from SI global" ) # converting 1 s in min
    ok &= EXPECT_FLOAT_EQ( 23.12, units.time_from_SI( units.time_to_SI( 23.12 ) ), "time from_to SI global" ) # converting 23.12 s in min then in s


    ### DERIVATED

    # area
    ok &= EXPECT_FLOAT_EQ( 100*100, units.area_from_SI( 1 ), "area from SI global" ) # converting 1 m2 in cm2
    ok &= EXPECT_FLOAT_EQ( 23.12, units.area_from_SI( units.area_to_SI( 23.12 ) ), "area from_to SI global" ) # converting 23.12 m2 in cm2 then in m2

    # volume
    ok &= EXPECT_FLOAT_EQ( 100*100*100, units.volume_from_SI( 1 ), "volume from SI global" ) # converting 1 m3 in cm3
    ok &= EXPECT_FLOAT_EQ( 23.12, units.volume_from_SI( units.volume_to_SI( 23.12 ) ), "volume from_to SI global" ) # converting 23.12 m3 in cm3 then in m3


    # velocity
    ok &= EXPECT_FLOAT_EQ( 60000, units.velocity_from_SI( 1, units.length_mm, units.time_min ), "velocity from SI local" ) # converting 1 m/s in mm/min
    ok &= EXPECT_FLOAT_EQ( 6000, units.velocity_from_SI( 1 ), "velocity from SI global" ) # converting 1 m/s in cm/min
    ok &= EXPECT_FLOAT_EQ( 23.12, units.velocity_from_SI( units.velocity_to_SI( 23.12 ) ), "velocity from_to SI global" ) # converting 23.12 m/s in cm/min then in m/s

    # acceleration
    ok &= EXPECT_FLOAT_EQ( 100*3600.0, units.acceleration_from_SI( 1 ), "acceleration from SI global" ) # converting 1 m/s2 in cm/min2
    ok &= EXPECT_FLOAT_EQ( 23.12, units.acceleration_from_SI( units.acceleration_to_SI( 23.12 ) ), "acceleration from_to SI global" ) # converting 23.12 m/s2 in cm/min2 then in m/s2


    # force
    ok &= EXPECT_FLOAT_EQ( 1000*100*3600, units.force_from_SI( 1 ), "force from SI global" ) # converting 1 N in g.cm/min2
    ok &= EXPECT_FLOAT_EQ( 23.12, units.force_from_SI( units.force_to_SI( 23.12 ) ), "force from_to SI global" )  # converting 23.12 N in g.cm/min2 then in N

    # pressure
    ok &= EXPECT_FLOAT_EQ( 1000/100.0*3600, units.pressure_from_SI( 1 ), "pressure from SI global" ) # converting 1 Pa in g/(cm.min2)
    ok &= EXPECT_FLOAT_EQ( 23.12, units.pressure_from_SI( units.pressure_to_SI( 23.12 ) ), "pressure from_to SI global" ) # converting 23.12 Pa in g/(cm.min2) then in Pa

    # energy
    ok &= EXPECT_FLOAT_EQ( 1000*100*100*3600, units.energy_from_SI( 1 ), "energy from SI global" ) # converting 1 J in g.cm2/min2
    ok &= EXPECT_FLOAT_EQ( 23.12, units.energy_from_SI( units.energy_to_SI( 23.12 ) ), "energy from_to SI global" ) # converting 23.12 J in g.cm2/min2 then in J

    ### MATERIALS

    # elasticity
    ok &= EXPECT_FLOAT_EQ( 1000/100.0*3600, units.elasticity_from_SI( 1 ), "elasticity from SI global" ) # converting 1 Pa in g/(cm.min2)
    ok &= EXPECT_FLOAT_EQ( 23.12, units.elasticity_from_SI( units.elasticity_to_SI( 23.12 ) ), "elasticity from_to SI global" ) # converting 23.12 Pa in g/(cm.min2) then in Pa

    # density
    ok &= EXPECT_FLOAT_EQ( 1000/100.0/100.0/100.0, units.density_from_SI( 1 ), "density from SI global" ) # converting 1 kg/m3 in g/cm3
    ok &= EXPECT_FLOAT_EQ( 23.12, units.density_from_SI( units.density_to_SI( 23.12 ) ), "density from_to SI global" ) # converting 23.12 kg/m3 in g/cm3 then in kg/m3


    ### MECHANICS

    # stress
    ok &= EXPECT_FLOAT_EQ( 1000/100.0*3600, units.stress_from_SI( 1 ), "stress from SI global" ) # converting 1 Pa in g/(cm.min2)
    ok &= EXPECT_FLOAT_EQ( 23.12, units.stress_from_SI( units.pressure_to_SI( 23.12 ) ), "stress from_to SI global" ) # converting 23.12 Pa in g/(cm.min2) then in Pa

    # deformation
    ok &= EXPECT_FLOAT_EQ( 100, units.deformation_from_SI( 1 ), "deformation from SI global" ) # converting 1 m in cm
    ok &= EXPECT_FLOAT_EQ( 23.12, units.deformation_from_SI( units.length_to_SI( 23.12 ) ), "deformation from_to SI global" ) # converting 23.12 m in cm then in m



    return ok

