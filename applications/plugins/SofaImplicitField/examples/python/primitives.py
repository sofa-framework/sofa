
from SofaImplicitField import ScalarField
import numpy

class Sphere(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)

        self.addData("center", type="Vec3d",value=kwargs.get("center", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self.addData("radius", type="double",value=kwargs.get("radius", 1.0), default=1, help="radius of the sphere", group="Geometry")

    def getValue(self, pos):
        x,y,z = pos
        return numpy.sqrt( numpy.sum((self.center.value - numpy.array([x,y,z]))**2) ) - self.radius.value 