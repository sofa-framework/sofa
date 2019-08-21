# -*- coding: utf-8 -*-
# This file requires gmshpy to be installed. 
# Author: stefan.escaida-navarro@inria.fr
def tetmeshFromBrepAndSaveToFile(filepath, outputdir='autogen', **kwargs):
        """generate a tetrahedron mesh from the provided file and store the 
           result in a vtk file. The filename is returned. 
           
           :param str filepath:
           :param str outputdir:
           :param float Mesh_CharacteristicLengthFactor:
           :param float Mesh_CharacteristicLengthMax:
           :param float Mesh_CharacteristicLengthMin: 
           :param float View_GeneralizedRaiseZ
        """ 
        import gmshpy
        import hashlib
        import os
        import numpy as np
        import time
        import locale
        locale.setlocale(locale.LC_ALL, 'C')

        # Set options from kwargs
        ArgumentsStrings = []
        ValuesStrings = []
        for key, value in kwargs.items():
            OptionString = key
            OptionValue = value
            SplitStr = OptionString.split('_')
            Category = SplitStr[0]
            Option = SplitStr[1]
            if isinstance(OptionValue, basestring):  # need to be careful to call the correct function 
                gmshpy.GmshSetStringOption(Category, Option, OptionValue)
            else:
                gmshpy.GmshSetNumberOption(Category, Option, OptionValue)
            #Warning: these functions return no value to indicate success of setting an option!
            ArgumentsStrings.append(OptionString)
            ValuesStrings.append(OptionValue)

        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)

        FilePathSplit = filepath.split('/')
        FileName = FilePathSplit[-1] 
        FileNameSplit = FileName.split('.')
        FileNameNoExtension = FileNameSplit[-1]

        SortingIdxs = np.argsort(ArgumentsStrings)

        # Hashing
        ParametricGeometryFile = open(filepath)
        # Warning: here we are not taking into account that the file could use a large amount of memory
        FileContents = ParametricGeometryFile.read()
        
        # hash the file contents
        FileAndOptionsHashObj = hashlib.sha256(FileContents)
        
        # add the options strings to the hash
        for i in SortingIdxs:
            ArgsForHash = ArgumentsStrings[i] + '=' + str(ValuesStrings[i]) + ';'
            FileAndOptionsHashObj.update(ArgsForHash)

        HashStr = FileAndOptionsHashObj.hexdigest()
        
        outfilepath = os.path.join(outputdir, HashStr + '.vtk')
        if os.path.exists(outfilepath):
            print('Find a file with an identical hash. Returning from cache.')                
            return outfilepath
                    
        #generate
        print('Beginning meshing: ')
        GeometricModel = gmshpy.GModel()
        GeometricModel.load(filepath)
        GeometricModel.mesh(3)
        GeometricModel.save(outfilepath)
        print('Finished meshing.')

        #HashFile = open(HashFilePath, 'w+')
        #HashFile.write(HashStr+'\n')
        #HashFile.write('# Options:\n')
        #for i in SortingIdxs:
        #        HashFile.write('# ' + ArgumentsStrings[i]+'='+str(ValuesStrings[i])+';\n')
        #HashFile.close()

        return outfilepath

def createScene(root):
        from stlib.scene import Scene

        Scene(root)
        root.VisualStyle.displayFlags="showForceFields"

        filename = tetmeshFromBrepAndSaveToFile(filepath='data/meshes/CapNoCavities.brep', 
                                      outputdir='data/meshes/autogen/',
                                      Mesh_CharacteristicLengthFactor=0.4, 
                                      Mesh_CharacteristicLengthMax=3, 
                                      Mesh_CharacteristicLengthMin=0.1, 
                                      View_GeneralizedRaiseZ='v0')
                                      
        root.createObject("MeshVTKLoader", name="loader", filename=filename)
        root.createObject("TetrahedronSetTopologyContainer", name="container", src="@loader")

        root.createObject("MechanicalObject", name="dofs", position="@loader.position")
        root.createObject("TetrahedronFEMForceField", name="forcefield")                              

