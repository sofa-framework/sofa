from gmesh import tetmeshFromBrepAndSaveToFile

from pygmsh.built_in import Geometry 
def meshAndSaveToFile(geom, directory="autogen"):                    
        import meshio
        import md5
        import os
        import pygmsh

        code = geom.get_code()
        m = md5.new()
        m.update(code)
        filename = os.path.join(directory, m.hexdigest()+".vtk")
        if not os.path.exists(directory):
                os.makedirs(directory)        
        if not os.path.exists(filename):     
                points, cells, point_data, cell_data, field_data = pygmsh.generate_mesh(geom)
                meshio.write_points_cells(filename, points, cells, cell_data=cell_data)
        return filename

def createScene(root):
        from math import pi
        geom = Geometry()
        
        # Draw a cross.
        poly = geom.add_polygon([
            [ 0.1,  0.5, 0.0],
            [-0.1,  0.1, 0.0],
            [-0.5,  0.0, 0.0],
            [-0.1, -0.1, 0.0],
            [ 0.0, -0.5, 0.0],
            [ 0.1, -0.1, 0.0],
            [ 0.5,  0.0, 0.0],
            [ 0.1,  0.1, 0.0]
            ],
            lcar=0.05
            )
        axis = [0, 0, 1]

        geom.extrude(
            poly,
            translation_axis=axis,
            rotation_axis=axis,
            point_on_axis=[0, 0, 0],
            angle=2.0 / 6.0 * pi
            )

        filename = meshAndSaveToFile(geom, directory="data/meshes/autogen/")
        root.createObject("MeshVTKLoader", name="loader", filename=filename)
        root.createObject("TetrahedronSetTopologyContainer", name="container", src="@loader")

        root.createObject("MechanicalObject", name="dofs", position="@loader.position")
        root.createObject("TetrahedronFEMForceField", name="forcefield")
