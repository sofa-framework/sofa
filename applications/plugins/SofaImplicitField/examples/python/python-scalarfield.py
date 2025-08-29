import Sofa
from SofaImplicitField import ScalarField
from SofaTypes.SofaTypes import Vec3d, Mat3x3
import numpy

class Sphere(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        
        self.addData("center", type="Vec3d",value=kwargs.get("center", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self.addData("radius", type="double",value=kwargs.get("radius", 1.0), default=1, help="radius of the sphere", group="Geometry")

    def getValue(self, position):
        x,y,z = position
        return numpy.sqrt( numpy.sum((self.center.value - numpy.array([x,y,z]))**2) ) - self.radius.value 

class SphereWithCustomHessianAndGradient(Sphere):
    def __init__(self, *args, **kwargs):
        Sphere.__init__(self, *args, **kwargs)#

    def getGradient(self, position):
        return Vec3d(3.0,2.0,1.0)

    def getHessian(self, position):
        return Mat3x3([[1,1,1],[2,1,1],[3,1,1]])

class FieldController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.field = kwargs.get("target")

    def onAnimateEndEvent(self, event):
        print("Animation end event, ")
        print("Field value at 0,0,0 is: ", self.field.getValue(Vec3d(0.0,0.0,0.0)) )
        print("Field value at 1,0,0 is: ", self.field.getValue(Vec3d(1.0,0.0,0.0)) )
        print("Field value at 2,0,0 is: ", self.field.getValue(Vec3d(2.0,0.0,0.0)) )

        print("Gradient value at 0,0,0 is: ", type(self.field.getGradient(Vec3d(0.0,0.0,0.0))))
        print("Hessian value at 0,0,0 is: ", type(self.field.getHessian(Vec3d(0.0,0.0,0.0))))

def createScene(root):
    """In this scene we create two scalar field of spherical shape, the two are implemented using
    python. The first one is overriding only the getValue, the hessian and gradient is thus computed using
    finite difference in the c++ code. The second field is overriding the hessian and gradient function
    """
    root.addObject(Sphere("field1"))
    root.addObject(FieldController(name="controller1", target=root.field1))

    root.addObject(SphereWithCustomHessianAndGradient("field2"))
    root.addObject(FieldController(name="controller2", target=root.field2))
