"""Distance field function

Sources:
   https://iquilezles.org/articles/distfunctions/
"""
from SofaImplicitField import ScalarField
import numpy

class Sphere(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)

        self.addData("center", type="Vec3d",value=kwargs.get("center", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self.addData("radius", type="double",value=kwargs.get("radius", 1.0), default=1, help="radius of the sphere", group="Geometry")

    def getValue(self, pos):
        x,y,z = pos
        return numpy.linalg.norm(self.center.value - numpy.array([x,y,z])) - self.radius.value

class RoundedBox(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)

        self.addData("center", type="Vec3d",value=kwargs.get("center", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self.addData("dimensions", type="Vec3d",value=kwargs.get("dimensions", [1.0,1.0,1.0]), default=[1.0,1.0,1.0], help="dimmension of the box", group="Geometry")
        self.addData("rounding_radius", type="double",value=kwargs.get("rounding_radius", 0.1), default=0.1, help="radius of the sphere", group="Geometry")

    def getValue(self, pos):
        x,y,z = pos
        b = self.dimensions.value
        r = self.rounding_radius.value
        q = numpy.abs(self.center.value - numpy.array([x,y,z])) - b + r
        res = numpy.linalg.norm(numpy.maximum(q, 0.0)) + min(max(q[0], max(q[1],q[2]) ), 0.0) - r
        return res
