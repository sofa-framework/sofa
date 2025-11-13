from SofaImplicitField import ScalarField
import numpy

class Union(ScalarField):
    """Union of two scalar fields"""
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)

        self.childA = kwargs.get("childA", None)
        self.childB = kwargs.get("childB", None)

    def getValue(self, position):
        return min(self.childA.getValue(position), self.childB.getValue(position))

class Difference(ScalarField):
    """Difference of two scalar fields"""
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)

        self.childA = kwargs.get("childA", None)
        self.childB = kwargs.get("childB", None)

    def getValue(self, position):
        return max(-self.childA.getValue(position), self.childB.getValue(position))

class Intersection(ScalarField):
    """Intersection of two scalar fields"""
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)

        self.childA = kwargs.get("childA", None)
        self.childB = kwargs.get("childB", None)

    def getValue(self, position):
        return max(self.childA.getValue(position), self.childB.getValue(position))