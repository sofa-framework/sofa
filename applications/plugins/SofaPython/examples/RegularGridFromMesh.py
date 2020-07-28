import argparse
import Sofa
import sys
import os
import math
import numpy as np

def createGrid(root, x, y, z, mesh, name):
    root.createObject('MeshObjLoader', name="loader", filename=mesh)
    model = root.createObject('OglModel', name="model", src="@loader")
    model.init()
    model.computeBBox()
        
    grid = root.createObject('RegularGrid', name="grid", nx=x, ny=y, nz=z, min=model.bbox.minBBox, max=model.bbox.maxBBox)
    root.createObject('MechanicalObject', name="DOFs")

    topoExporter = root.createChild('Topo')
    topoContainer = topoExporter.createObject('TetrahedronSetTopologyContainer')
    topoExporter.createObject('TetrahedronSetTopologyModifier')
    topoExporter.createObject('TetrahedronSetTopologyAlgorithms', template="Vec3d")
    topoExporter.createObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", drawTetrahedra="0")
    topoExporter.createObject('Hexa2TetraTopologicalMapping', name="default28", input=grid.getLinkPath(),
                              output=topoContainer.getLinkPath())
    # topoExporter.createObject('VTKExporter', position="@../DOFs.position", edges="0", tetras="1",
    #                           filename=self.name, exportAtBegin="true")

    root.createObject('TetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio="0.4", method="large")
        
def createScene(node):
    if len (sys.argv) > 1:
        sys.argv = sys.argv[1].split()

    parser = argparse.ArgumentParser(
        description='Loads a 3D model a generates a regular grid using its bounding box')
    parser.add_argument(
        '-x','--gridx', dest='nx', type=int, default=10, help='')
    parser.add_argument(
        '-y','--gridy', dest='ny', type=int, default=10, help='')
    parser.add_argument(
        '-z', '--gridz', dest='nz', type=int, default=10, help='')
    parser.add_argument(
        '-m', '--mesh', dest='mesh', type=str, default="mesh/liver.obj", help='')
    parser.add_argument(
        '-o', '--output', dest='name', type=str, default="liverGrid", help='')
    args, unknown = parser.parse_known_args(sys.argv)
    createGrid(node, args.nx, args.ny, args.nz, args.mesh, args.name)
    return


