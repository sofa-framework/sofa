import Sofa
from primitives import Sphere

class FieldController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.field = kwargs.get("target")

    def onAnimateEndEvent(self, event):
        print("Animation end event")
        print("Field value at 0,0,0 is: ", self.field.getValue([0.0,0.0,0.0]) )
        print("Field value at 1,0,0 is: ", self.field.getValue([1.0,0.0,0.0]) )
        print("Field value at 2,0,0 is: ", self.field.getValue([2.0,0.0,0.0]) )

def createScene(root):
    root.addObject(Sphere("field"))
    root.addObject(FieldController(target=root.field))
