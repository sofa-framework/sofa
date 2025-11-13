import Sofa
from Sofa.Types import RGBAColor 
from xshape.primitives import *
from xshape.transforms import *
from xshape.operators import *  

class DrawController(Sofa.Core.Controller):
   def __init__(self, *args, **kwargs):
      Sofa.Core.Controller.__init__(self, *args, **kwargs)

   def draw(self, visual_context):
      dt = visual_context.getDrawTool()
      dt.drawText([-1.0, 1.0, 0.5], 0.2, "Union(Sphere, Box)", RGBAColor(1.0,1.0,1.0,1.0))
      dt.drawText([ 1.0, 1.0, 0.5], 0.2, "Difference(Sphere, Box)", RGBAColor(1.0,1.0,1.0,1.0))
      
def createScene(root : Sofa.Core.Node):
    """Creates two different mesh from two scalar field.
       The scalar fields are 'spherical', one implemented in python, the other in c++
       One of the produced mesh is then connected to a visual model.
    """
    root.addObject("RequiredPlugin", name="SofaImplicitField")

    root.addObject(DrawController())

    ########################### Fields ##################
    root.addChild("Fields")
    f1 = root.Fields.addObject(
      Union(name="field1",
            childA=Sphere(name="sphere", center=[0,0,0],radius=0.7),
            childB=RoundedBox(center=[0.0,0.0,0.0],dimensions=[0.95,0.5,0.5], rounding_radius=0.1))
   )

    f2 = root.Fields.addObject(
      Difference(name="field2",
            childB=Sphere(name="sphere", center=[2,0,0],radius=0.9),
            childA=RoundedBox(center=[2.0,0.0,0.0],dimensions=[0.95,0.5,0.5], rounding_radius=0.1))
   )
    
    ########################### Meshing ##################
    root.addChild("Meshing")
    m1 = root.Meshing.addObject("FieldToSurfaceMesh", name="polygonizer1",
                          field=f1.linkpath, min=[-1,-1,-1], max=[1,1,1],
                          step=0.1, debugDraw=True)

    m2 = root.Meshing.addObject("FieldToSurfaceMesh", name="polygonizer2",
                          field=f2.linkpath, min=[1,-1,-1], max=[3,1,1],
                          step=0.07)

    ########################### Fields ##################
    root.addChild("Visual")
    root.Visual.addObject("OglModel", name="renderer",
                        position=root.Meshing.polygonizer2.points.linkpath,
                        triangles=root.Meshing.polygonizer2.triangles.linkpath)
