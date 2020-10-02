## testing RigidMassInfo
# a difficulty is to test the orientatoin of the summed up rigid frame since it is not deterministic (axis swapping is expected)
# it is indirectly tested with test where the resultant sum is symmetric (a cube) so that the inertia is equal on all axis

import os
import numpy

from SofaTest.Macro import *

from SofaPython import Quaternion
from SofaPython import mass

def run():
    ok=True

    # cube_1, size 1, one corner at origin
    cubeMass_1 = mass.RigidMassInfo()
    cubeMass_1.mass = 1.
    cubeMass_1.com=[0.5,0.5,0.5]
    cubeMass_1.diagonal_inertia=[1./6.,1./6.,1./6.]
    cubeMass_1.density = 1.5

    # cube_2, half cube, along x axis, positive side
    cubeMass_2 = mass.RigidMassInfo()
    cubeMass_2.mass = 0.5
    cubeMass_2.com=[0.25,0.,0.]
    cubeMass_2.diagonal_inertia=(0.5/12.)*numpy.array([2.,1.25,1.25])
    cubeMass_2.density = 1.

    # cube_3, half cube, along x axis, negative side
    cubeMass_3 = mass.RigidMassInfo()
    cubeMass_3.mass = 0.5
    cubeMass_3.com=[-0.25,0.,0.]
    cubeMass_3.diagonal_inertia=cubeMass_2.diagonal_inertia
    cubeMass_3.density = 2.

    ok &= EXPECT_MAT_EQ([[2./3., -0.25, -0.25],[-0.25,2./3.,-0.25],[-0.25,-0.25,2./3.]],
                        cubeMass_1.getWorldInertia(),
                        "RigidMassInfo.getWorldInertia() cube 1")

    cubeMass_1_1 = cubeMass_1+cubeMass_1
    ok &= EXPECT_FLOAT_EQ(2., cubeMass_1_1.mass, "RigidMassInfo.add cube_1+cube_1 - mass")
    ok &= EXPECT_FLOAT_EQ(cubeMass_1.density, cubeMass_1_1.density, "RigidMassInfo.add cube_1+cube_1 - density")
    ok &= EXPECT_VEC_EQ([0.5,0.5,0.5], cubeMass_1_1.com, "RigidMassInfo.add  cube_1+cube_1 - com")
    ok &= EXPECT_MAT_EQ([1./3.,1./3.,1./3.], cubeMass_1_1.diagonal_inertia, "RigidMassInfo.add cube_1+cube_1 - diagonal_inertia" )

    cubeMass_2_3 = cubeMass_2+cubeMass_3
    ok &= EXPECT_FLOAT_EQ(1., cubeMass_2_3.mass, "RigidMassInfo.add cube_2+cube_3 - mass")
    ok &= EXPECT_FLOAT_EQ(1.+1./3., cubeMass_2_3.density, "RigidMassInfo.add cube_2+cube_3 - density")
    ok &= EXPECT_VEC_EQ([0.,0.,0.], cubeMass_2_3.com, "RigidMassInfo.add cube_2+cube_3 - com")
    ok &= EXPECT_MAT_EQ([1./6.,1./6.,1./6.], cubeMass_2_3.diagonal_inertia, "RigidMassInfo.add cube_2+cube_3 - diagonal_inertia" )

    # modif cube 2 and 3 to be rotated around z axis
    qq = [ Quaternion.axisToQuat([0.,0.,1.],math.radians(30)),
           Quaternion.axisToQuat([0.,-2.,1.],math.radians(-60)),
           Quaternion.axisToQuat([-3.,2.,-1.],math.radians(160)) ]
    for q in qq:
        cubeMass_2.com=Quaternion.rotate(q, [0.25,0.,0.])
        cubeMass_2.inertia_rotation=Quaternion.inv(q)
        cubeMass_3.com=Quaternion.rotate(q, [-0.25,0.,0.])
        cubeMass_3.inertia_rotation=Quaternion.inv(q)

        cubeMass_2_3 = cubeMass_2+cubeMass_3
        ok &= EXPECT_FLOAT_EQ(1., cubeMass_2_3.mass, "RigidMassInfo.add rotated cube_2+cube_3 - mass")
        ok &= EXPECT_VEC_EQ([0.,0.,0.], cubeMass_2_3.com, "RigidMassInfo.add rotated cube_2+cube_3 - com")
        ok &= EXPECT_MAT_EQ([1./6.,1./6.,1./6.], cubeMass_2_3.diagonal_inertia, "RigidMassInfo.add rotated cube_2+cube_3 - diagonal_inertia" )

    return ok
