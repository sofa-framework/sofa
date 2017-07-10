## testing Rigid.generate_rigid that computes com, mass and inertia from a mesh

# TODO compute and test principal axes basis by rotating meshes


from SofaTest.Macro import *
from Compliant import Tools
from SofaPython import Quaternion
import SofaPython.mass
import numpy

path = Tools.path( __file__ ) + "/geometric_primitives/"


# cube.obj is unit 1x1x1 cube
# sphere.obj radius=1
# cylinder.obj, height=1 in z, radius=1
# all meshes are centered in (0.5,0.5,0.5)

meshes = ['cube.obj', 'sphere.obj', 'cylinder.obj']
scales = [[1,1,1],[1,1,10],[3.3,3.3,10],[10,5,2]]
densities = [1, 1000, 7.7]
rotations = [[90,0,0],[22.456,0,0],[0,90,0],[0,23.546,0],[0,0,90],[0,0,-63.2],[90,90,0],[90,12.152,0],[25.645,12.36,0],[90,90,90],[-12.356,124.33,-56.1]]

# prepare multi-dimensional structure
masses = []
for x in range(len(meshes)):
    row = []
    for y in xrange(len(scales)):
        col = []
        for z in xrange(len(densities)):
            col.append(0)
        row.append(col)
    masses.append(row)

# theoretical masses
for s in xrange(len(scales)):
    scale = scales[s]
    for d in xrange(len(densities)):
        density = densities[d]
        masses[0][s][d] = density * scale[0]*scale[1]*scale[2] # cuboid
        masses[1][s][d] = density * 4.0/3.0*math.pi * scale[0]*scale[1]*scale[2]  # ellispoid
        masses[2][s][d] = density * math.pi * scale[0]*scale[1]*scale[2] # cylinder-like (ellipse area)


# prepare multi-dimensional structure
inertia = []
for x in range(len(meshes)):
    row = []
    for y in xrange(len(scales)):
        col = []
        for z in xrange(len(densities)):
            third = []
            for w in xrange(3):
                third.append(0)
            col.append(third)
        row.append(col)
    inertia.append(row)

# theoretical diagonal inertia (along principal axes)
for s in xrange(len(scales)):
    scale = scales[s]
    for d in xrange(len(densities)):
        density = densities[d]

        #cuboid
        mass = masses[0][s][d]
        inertia[0][s][d][0] = 1.0/12.0 * mass * (scale[1]*scale[1]+scale[2]*scale[2]) # x
        inertia[0][s][d][1] = 1.0/12.0 * mass * (scale[0]*scale[0]+scale[2]*scale[2]) # y
        inertia[0][s][d][2] = 1.0/12.0 * mass * (scale[0]*scale[0]+scale[1]*scale[1]) # z

        #ellipsoid
        mass = masses[1][s][d]
        inertia[1][s][d][0] = 1.0/5.0 * mass * (scale[1]*scale[1]+scale[2]*scale[2]) # x
        inertia[1][s][d][1] = 1.0/5.0 * mass * (scale[0]*scale[0]+scale[2]*scale[2]) # y
        inertia[1][s][d][2] = 1.0/5.0 * mass * (scale[0]*scale[0]+scale[1]*scale[1]) # z

        #cylinder   WARNING FALSE for scale[0]!=scale[1] ie ellipse area
        mass = masses[2][s][d]
        inertia[2][s][d][0] = 1.0/12.0 * mass * (3*scale[0]*scale[1]+scale[2]*scale[2]) # x
        inertia[2][s][d][1] = 1.0/12.0 * mass * (3*scale[0]*scale[1]+scale[2]*scale[2]) # y
        inertia[2][s][d][2] = 1.0/2.0 * mass * scale[0]*scale[1] # z


def almostEqualReal( a, b, epsilon = 1e-1 ): # really poor precision for mesh-based
    ## compare two reals for a given relative precision
    epsilon = math.fabs( epsilon * a ) # relative precision...
    return math.fabs( a - b ) < epsilon

def almostEqualLists( a, b, epsilon = 1e-1 ):
    ## compare two lists of reals for a given relative precision
    if len(a)!=len(b):
        return False
    for x in xrange(len(a)):
        if not almostEqualReal(a[x],b[x],epsilon):
            return False;
    return True




def run():

    ok = True

    info = SofaPython.mass.RigidMassInfo()

# testing axis-aligned known geometric shapes
    for m in xrange(len(meshes)):
        mesh = meshes[m]
        mesh_path = path + meshes[m]

        for s in xrange(len(scales)):
            scale = scales[s]

            if mesh=="cylinder.obj" and scale[0]!=scale[1]:
                continue
        
            for d in xrange(len(densities)):
                density=densities[d]

                info.setFromMesh( mesh_path, density, scale )

                error = " ("+meshes[m]+", s="+Tools.cat(scale)+" d="+str(density)+")"

                ok &= EXPECT_TRUE( almostEqualReal(info.mass, masses[m][s][d]), "mass"+error+" "+str(info.mass)+"!="+str(masses[m][s][d]) )
                ok &= EXPECT_TRUE( almostEqualLists(info.com,[x*0.5 for x in scale]), "com"+error+" "+Tools.cat(info.com)+"!="+Tools.cat([x*0.5 for x in scale]) )
                ok &= EXPECT_TRUE( almostEqualLists(info.diagonal_inertia,inertia[m][s][d]), "inertia"+error+" "+str(info.diagonal_inertia)+"!="+str(inertia[m][s][d]) )

# testing diagonal inertia extraction from a rotated cuboid
    mesh = "cube.obj"
    mesh_path = path + mesh
    scale = scales[3]
    density = 1
    theory = sorted(inertia[0][3][0])
    for r in rotations:
        info.setFromMesh( mesh_path, density, scale, r )
        local = sorted(info.diagonal_inertia)
        ok &= EXPECT_TRUE( almostEqualLists(local,theory), "inertia "+str(local)+"!="+str(theory)+" (rotation="+str(r)+")" )

# testing extracted inertia rotation
    mesh = "rotated_cuboid_12_35_-27.obj"
    mesh_path = path + mesh
    density = 1
    info.setFromMesh( mesh_path, density )

    # theoretical results
    scale = [2,3,1]
    mass = density * scale[0]*scale[1]*scale[2]
    inertiat = numpy.empty(3)
    inertiat[0] = 1.0/12.0 * mass * (scale[1]*scale[1]+scale[2]*scale[2]) # x
    inertiat[1] = 1.0/12.0 * mass * (scale[0]*scale[0]+scale[2]*scale[2]) # y
    inertiat[2] = 1.0/12.0 * mass * (scale[0]*scale[0]+scale[1]*scale[1]) # z

    # used quaternion in mesh

    q = Quaternion.normalized( Quaternion.from_euler( [12*math.pi/180.0, 35*math.pi/180.0, -27*math.pi/180.0] ) )

    # corresponding rotation matrices (ie frame defined by columns)
    mt = Quaternion.to_matrix( q )
    m  = Quaternion.to_matrix( info.inertia_rotation )

    # matching inertia
    idxt = numpy.argsort(inertiat)
    idx  = numpy.argsort(info.diagonal_inertia)

    # checking if each axis/column are parallel (same or opposite for unitary vectors)
    for i in xrange(3):
        ok &= EXPECT_TRUE( almostEqualLists(mt[:,idxt[i]].tolist(),m[:,idx[i]].tolist(),1e-5) or almostEqualLists(mt[:,idxt[i]].tolist(),(-m[:,idx[i]]).tolist(),1e-5), "wrong inertia rotation" )


#    print mt[:,idxt]
#    print m [:,idx ]


    return ok
