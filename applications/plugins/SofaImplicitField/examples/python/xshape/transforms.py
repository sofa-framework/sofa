from SofaImplicitField import ScalarField
import numpy

class Translate(ScalarField):
    """Translate a scalar field given as attribute"""
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)

        self.addData("translate", type="Vec3d",value=kwargs.get("translate", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="amount of translation", group="Geometry")
        self.child = kwargs.get("child", None)

    def getValue(self, pos):
        x,y,z = pos
        position = numpy.array([x,y,z])-self.translate.value
        return self.child.getValue(position)