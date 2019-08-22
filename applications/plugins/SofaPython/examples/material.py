import SofaPython.Tools

material = SofaPython.Tools.Material()
material.load("material.json")

print "wood:", material.density("wood"), "unknown:", material.density("unknown")

def createScene(rootNode):
    pass