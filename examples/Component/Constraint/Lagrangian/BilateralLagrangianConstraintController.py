import Sofa
from Sofa.constants import Key

class BilateralConstraintController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.constraint = kwargs.get("constraint") 

    def onKeyreleasedEvent(self, event):
        if event["key"] == Key.A:
            self.constraint.activate.value = not self.constraint.activate.value
             
def createScene(root):
    root.addObject("MechanicalObject")
    root.addObject("BilateralLagrangianConstraint", name="c")
    root.addObject(BilateralConstraintController(constraint=root.c))