import Sofa
from Sofa.constants import Key
from splib3.loaders.xmlloader import loadXML

class BilateralConstraintController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.constraints = kwargs.get("constraints", [])

    def onKeyreleasedEvent(self, event):
        if event["key"] == Key.A:
            for constraint in self.constraints:
                constraint.activate.value = not constraint.activate.value
             
def createScene(root):
    loadXML("BilateralInteractionConstraint_NNCG.scn", root)
    listConstraints = [ object for object in root.objects if object.getClassName() == "BilateralLagrangianConstraint"]
    root.addObject(BilateralConstraintController(constraints=listConstraints))