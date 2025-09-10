import Sofa
from SofaTypes import Vec3d

class MyController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self,*args,**kwargs)

    def draw(self, params):
        print("DRAW", type(params))
        dt = params.getDrawTool()
        dt.drawPoints([Vec3d(0,0,0), Vec3d(1,0,0)], 10)

def createScene(root):
    root.addObject(MyController(name="controller"))