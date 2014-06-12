## testing Rigid.generate_rigid that computes com, mass and inertia from a mesh

# TODO compute and test principal axes basis by rotating meshes


from SofaTest.Macro import *
from Compliant import Rigid, Tools

path = Tools.path( __file__ ) + "/geometric_primitives/"


# cube.obj is unit 1x1x1 cube
# sphere.obj radius=1
# cylinder.obj, height=1 in z, radius=1
# all meshes are centered in (0.5,0.5,0.5)

meshes = ['cube.obj', 'sphere.obj', 'cylinder.obj']
scales = [[1,1,1],[1,1,10],[3.3,3.3,10],[10,5,2]]
densities = [1, 1000, 7.7]

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
    ## compre two reals for a given relative precision
    epsilon = epsilon * a # relative precision...
    return math.fabs( a - b ) < epsilon

def almostEqualLists( a, b, epsilon = 1e-1 ):
    ## compre two lists of reals for a given relative precision
    if len(a)!=len(b):
        return False
    for x in xrange(len(a)):
        if not almostEqualReal(a[x],b[x],epsilon):
            return False;
    return True




def run():

    ok = True

    for m in xrange(len(meshes)):
        mesh = meshes[m]
        mesh_path = path + meshes[m]

        for s in xrange(len(scales)):
            scale = scales[s]

            if mesh=="cylinder.obj" and scale[0]!=scale[1]:
                continue
        
            for d in xrange(len(densities)):
                density=densities[d]

                info = Rigid.generate_rigid( mesh_path, density, scale )

                error = " ("+meshes[m]+", s="+Tools.cat(scale)+" d="+str(density)+")"

                ok &= EXPECT_TRUE( almostEqualReal(info.mass, masses[m][s][d]), "mass"+error+" "+str(info.mass)+"!="+str(masses[m][s][d]) )
                ok &= EXPECT_TRUE( almostEqualLists(info.com,[x*0.5 for x in scale]), "com"+error+" "+Tools.cat(info.com)+"!="+Tools.cat([x*0.5 for x in scale]) )
                ok &= EXPECT_TRUE( almostEqualReal(info.inertia[0],inertia[m][s][d][0]) and almostEqualReal(info.inertia[4], inertia[m][s][d][1]) and almostEqualReal(info.inertia[8], inertia[m][s][d][2]), "inertia"+error+" "+str([info.inertia[0],info.inertia[4],info.inertia[8]])+"!="+str(inertia[m][s][d]) )


    return ok
