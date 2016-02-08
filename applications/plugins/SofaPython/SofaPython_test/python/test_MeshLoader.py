import SofaPython.MeshLoader

from SofaTest.Macro import *

import sys

def run():

    objfile = SofaPython.Tools.localPath( __file__, "../../examples/data/dragon.obj" )

    objmesh = SofaPython.MeshLoader.loadOBJ( objfile )



    ok = True

    ok &= EXPECT_EQ( len(objmesh.vertices), 1190 )
    ok &= EXPECT_EQ( len(objmesh.normals), 0 )
    ok &= EXPECT_EQ( len(objmesh.faceVertices), 2564 )

    return ok