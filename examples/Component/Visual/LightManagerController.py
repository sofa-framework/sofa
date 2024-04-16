import Sofa
from Sofa.constants import Key
from splib3.loaders.xmlloader import loadXML

class LightManagerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.manager = kwargs.get("manager")

    def onKeyreleasedEvent(self, event):
        if event["key"] == Key.L:
            self.manager.shadows.value = not self.manager.shadows.value
            print("LIGHT .. ON/OFF ", self.manager.shadows.value)

def createScene(root):
    root = loadXML("OglShadowShader_Directional.scn", root)
    root.addObject(LightManagerController(manager=root.lightManager1))
