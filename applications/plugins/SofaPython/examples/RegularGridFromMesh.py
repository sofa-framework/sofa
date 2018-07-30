import argparse
import Sofa
import sys
import os
import math
import numpy as np

class CreateGrid(Sofa.PythonScriptController):

    mesh = "mesh/liver.obj"
    nx = 10
    ny = 10
    nz = 10
    
    def __init__(self, node, argv):
        try:
            parser = argparse.ArgumentParser(
                description='Loads a 3D model a generates a regular grid using its bounding box')
            parser.add_argument(
                '-x','--gridx', dest='nx', type=int, default=self.nx, help='')
            parser.add_argument(
                '-y','--gridy', dest='ny', type=int, default=self.ny, help='')
            parser.add_argument(
                '-z', '--gridz', dest='nz', type=int, default=self.nz, help='')
            parser.add_argument(
                '-m', '--mesh', dest='mesh', type=str, default=self.mesh, help='')
            parser.add_argument(
                '-o', '--output', dest='name', type=str, default="", help='')
            args = parser.parse_args(argv)
            
            self.nx = args.nx
            self.ny = args.ny
            self.nz = args.nz
            self.mesh = args.mesh
            self.name = args.name
        except:
            pass
        self.root = node
        self.createGraph(node)
        return None

    
    def createGraph(self,root):
        root.createObject('MeshObjLoader', name="loader", filename=self.mesh)
        model = root.createObject('OglModel', name="model", src="@loader")
        model.init()
        model.computeBBox()
        
        root.createObject('RegularGrid', name="grid", nx=self.nx, ny=self.ny, nz=self.nz,
                          min=model.bbox.minBBox, max=model.bbox.maxBBox)
        root.createObject('MechanicalObject', name="DOFs")

        topoExporter = root.createChild('Topo')
        topoExporter.createObject('TetrahedronSetTopologyContainer', name="Container")
        topoExporter.createObject('TetrahedronSetTopologyModifier')
        topoExporter.createObject('TetrahedronSetTopologyAlgorithms', template="Vec3d")
        topoExporter.createObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", drawTetrahedra="0")
        topoExporter.createObject('Hexa2TetraTopologicalMapping', name="default28", input="@../grid", output="@Container")
        # topoExporter.createObject('VTKExporter', position="@../DOFs.position", edges="0", tetras="1",
        #                           filename=self.name, exportAtBegin="true")

        root.createObject('TetrahedronFEMForceField', name="FEM", youngModulus=5000, poissonRatio="0.4", method="large")
        
def createScene(node):
    try :
        sys.argv[0]
    except :
        sys.argv = sys.argv[1].split()
    else:
        sys.argv = ['-h']

    pyController = CreateGrid(node,sys.argv)
    return


