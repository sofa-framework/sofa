import Sofa
from SofaImplicitField import ScalarField
import numpy

class Sphere(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        
        self.addData("center", type="Vec3d",value=kwargs.get("center", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self.addData("radius", type="double",value=kwargs.get("radius", 1.0), default=1, help="radius of the sphere", group="Geometry")

    def getValue(self, x, y, z):
        return numpy.sqrt( numpy.sum((self.center.value - numpy.array([x,y,z]))**2) ) - self.radius.value 

class FieldController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.field = kwargs.get("target")

    def onAnimateEndEvent(self, event):
        print("Animation end event")
        print("Field value at 0,0,0 is: ", self.field.getValue(0.0,0.0,0.0) ) 
        print("Field value at 1,0,0 is: ", self.field.getValue(1.0,0.0,0.0) ) 
        print("Field value at 2,0,0 is: ", self.field.getValue(2.0,0.0,0.0) ) 

def createScene(root):
    root.addObject(Sphere("field"))  
    root.addObject(FieldController(target=root.field))
