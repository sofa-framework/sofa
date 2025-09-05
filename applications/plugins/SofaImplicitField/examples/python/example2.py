import Sofa
from SofaImplicitField import ScalarField
import sys 

print("SYS : ", sys.path)
import numpy
import drjit  
from drjit.auto import Float, Array3f, UInt

#drjit.set_log_level(drjit.LogLevel.Info)


@drjit.freeze
def applyTranslation(position, translation, childFct):
    return childFct(position) 

class Translate(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        
        self.addData("translation", type="Vec3d",value=kwargs.get("translation", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self._translation = Array3f(self.translation.value)
        self.child = kwargs.get("child")
        self.opSdf = applyTranslation

    def sdf(self, positions):
        return self.child.sdf( positions - self._translation ) 

    def getValues(self, positions, out_values):
        """This version of the overrides get numpy array as input and output"""        
        positions = Array3f(positions[:, 0], positions[:, 1], positions[:, 2])
        t =  self.sdf(positions)
        out_values[:] = t 

class Difference(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        
        self.child1 = kwargs.get("child1")
        self.child2 = kwargs.get("child2")
        
    def sdf(self, position):
        c1 = self.child1.sdf(position)        
        c2 = self.child2.sdf(position)        
        return drjit.maximum(-c1, c2)

    def getValues(self, positions, out_values):
        """This version of the overrides get numpy array as input and output"""        
        positions = Array3f(positions[:, 0], positions[:, 1], positions[:, 2])
        t =  self.sdf(positions)
        out_values[:] = t 

@drjit.freeze
def drjit_length(vector):
    return drjit.sqrt( drjit.sum( vector**2 ) )

class Box(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        
        self.addData("b", type="Vec3d",value=kwargs.get("b", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")

        self._b = Array3f(self.b.value)

    def sdf(self, position):
        q = drjit.abs(position) - self._b        
        v = drjit.maximum(q, Array3f(0.0))    
        p = drjit.minimum(drjit.maximum(q.x, drjit.maximum(q.y, q.z)), Float(0.0))
        return drjit_length(v)

    def getValues(self, positions, out_values):
        """This version of the overrides get numpy array as input and output"""        
        positions = Array3f(positions[:, 0], positions[:, 1], positions[:, 2])
        t =  self.sdf(positions)
        out_values[:] = t 


@drjit.freeze
def sdfSphere(position, center, radius):
    return drjit.norm(position - center) - radius

class Sphere(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        
        self.addData("center", type="Vec3d",value=kwargs.get("center", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self.addData("radius", type="double",value=kwargs.get("radius", 1.0), default=1, help="radius of the sphere", group="Geometry")

        self._radius = Float(self.radius.value)
        self._center = Array3f(self.center.value)

    def sdf(self, positions):
        return sdfSphere(positions, self._center, self._radius)
        
    def getValue(self, position):
        x,y,z = position
        position = Array3f(x,y,z)
        t =  drjit.norm(position - self._center) - self._radius
        return t[0] 
    
    def getValues(self, positions, out_values):
        """This version of the overrides get numpy array as input and output"""        
        positions = Array3f(positions[:, 0], positions[:, 1], positions[:, 2])
        t =  self.sdf(positions)
        out_values[:] = t 

def createScene(root : Sofa.Core.Node):
    root.addObject("RequiredPlugin", name="SofaImplicitField")

    field1 = root.addObject(Sphere(name="field1", center=[0,0,0]))  
    field2 = root.addObject("SphericalField", center=[2,0,0])  

    translation1 = root.addObject(Translate(name="translate1", 
                                            translation=[-2.0,0,0], 
                                            child=field1))  

    field3 = root.addObject(Box(name="field3", b=[1.0,1.0,0.6]))  

    field4 = root.addObject(Difference(name="field4", child1=field1, child2=field3))  

    root.addChild("Visual")
 
    root.Visual.addObject("FieldToSurfaceMesh", name="polygonizer1",
                          field=field4.linkpath, min=[-1,-1,-1], max=[1,1,1],
                          isoValue="0.0", step="0.05", printLog=True, doAsync=False)    

    root.Visual.addObject("FieldToSurfaceMesh", name="polygonizer2",
                          field=field2.linkpath, min=[1,-1,-1], max=[3,1,1],
                          isoValue="0.0", step="0.05", printLog=True, doAsync=True)    

    root.Visual.addObject("FieldToSurfaceMesh", name="polygonizer3",
                          field=translation1.linkpath, min=[-4,-1,-1], max=[-1,1,1],
                          isoValue="0.0", step="0.1", printLog=True, doAsync=False)    

    #root.Visual.addObject("OglModel", name="renderer", 
    #                    position=root.Visual.polygonizer2.outputPoints.linkpath, 
    #                    triangles=root.Visual.polygonizer2.outputTriangles.linkpath)
    