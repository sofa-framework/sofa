import Sofa
from shapes import Sphere

def createScene(root : Sofa.Core.Node):
    root.addObject("RequiredPlugin", name="SofaImplicitField")
    
    s = root.addObject(Sphere(name="field1", center=[0,0,0]))  
    root.addObject("SphericalField", name="field2", center=[2,0,0])  

    root.addChild("Visual")
 
    root.Visual.addObject("FieldToSurfaceMesh", name="polygonizer1",
                          field=s.linkpath, min=[-1,-1,-1], max=[1,1,1],
                          isoValue="0.0", step="0.1",doAsync=True)    
 
    root.Visual.addObject("FieldToSurfaceMesh", name="polygonizer2",
                          field=root.field2.linkpath, min=[1,-1,-1], max=[3,1,1],
                          isoValue="0.0", step="0.01",doAsync=True)    
    
    root.Visual.addObject("OglModel", name="renderer", 
                        position=root.Visual.polygonizer2.outputPoints.linkpath, 
                        triangles=root.Visual.polygonizer2.outputTriangles.linkpath)
    