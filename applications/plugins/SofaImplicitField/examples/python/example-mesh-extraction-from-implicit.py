import Sofa
from primitives import Sphere

def createScene(root : Sofa.Core.Node):
    """Creates two different mesh from two scalar field.
       The scalar fields are 'spherical', one implemented in python, the other in c++
       One of the produced mesh is then connected to a visual model.
    """
    root.addObject("RequiredPlugin", name="SofaImplicitField")

    ########################### Fields ##################
    root.addChild("Fields")
    f1 = root.Fields.addObject(Sphere(name="field1", center=[0,0,0]))
    f2 = root.Fields.addObject("SphericalField", name="field2", center=[2,0,0])

    ########################### Meshing ##################
    root.addChild("Meshing")
    m1 = root.Meshing.addObject("FieldToSurfaceMesh", name="polygonizer1",
                          field=f1.linkpath, min=[-1,-1,-1], max=[1,1,1],
                          step=0.1, debugDraw=True)

    m2 = root.Meshing.addObject("FieldToSurfaceMesh", name="polygonizer2",
                          field=f2.linkpath, min=[1,-1,-1], max=[3,1,1],
                          step=0.01)

    ########################### Fields ##################
    root.addChild("Visual")
    root.Visual.addObject("OglModel", name="renderer",
                        position=root.Meshing.polygonizer2.points.linkpath,
                        triangles=root.Meshing.polygonizer2.triangles.linkpath)
