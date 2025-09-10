import Sofa
from SofaImplicitField import ScalarField
import numpy
import math

# float opTwist( in sdf3d primitive, in vec3 p )
# {
#     const float k = 10.0; // or some other amount
#     float c = cos(k*p.y);
#     float s = sin(k*p.y);
#     mat2  m = mat2(c,-s,s,c);
#     vec3  q = vec3(m*p.xz,p.y);
#     return primitive(q);
# }

class Twist(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        self.primitive = kwargs.get("primitive").getValue

        print("TEST", self.primitive(0.0,0.0,0.0))

    def getValue(self, x, y, z):
        # k = 10.0; 
        # c = math.cos(k*y)
        # s = math.sin(k*y)
        # m = numpy.array([[c,-s],[s,c]])
        # q = numpy.dot(m, numpy.array([x,z])) 
        # #value = self.primitive( q[0], q[1], y )
        value = self.primitive( x, y, z )
        # print("VALUE ", x, y, z, value )

        return value