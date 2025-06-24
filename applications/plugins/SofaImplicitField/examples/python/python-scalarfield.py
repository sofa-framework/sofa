import Sofa
from SofaImplicitField import ScalarField

class Sphere(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        self.addNewData("center")


    def getValue(self, x, y, z):
        return 123.456

def createScene(root):
    root.addObject(MyField("field"))  
    print( "Field value is: ", root.field.getValue(0.0,0.0,0.0) ) 