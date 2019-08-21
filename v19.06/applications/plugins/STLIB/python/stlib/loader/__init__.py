import Sofa
from splib.objectmodel import SofaPrefab, SofaObject

def addLoader(parent, filename,name="loader", translation=[0.0,0.0,0.0], eulerRotation=[0.0,0.0,0.0]):
        """Mesh Loader Prefab

           Use the file extension to create the adequate loader and add it to the scene.
        """
        if len(filename) == 0:
            raise Exception("Unable to instanciate the prefab Loader because the filename is empty")

        loader = None
        if filename.endswith(".msh"):
            return parent.createObject('MeshGmshLoader', name=name, filename=filename, rotation=eulerRotation, translation=translation)
        elif filename.endswith(".gidmsh"):
            return parent.createObject('GIDMeshLoader', name=name, filename=filename, rotation=eulerRotation, translation=translation)
        elif filename.endswith(".vtu") or filename.endswith(".vtk"):
            return parent.createObject('MeshVTKLoader', name=name, filename=filename, rotation=eulerRotation, translation=translation)

        raise Exception("Unable to instanciate the prefab Loader because the file extension is not supported")
