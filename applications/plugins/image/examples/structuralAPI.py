import SofaPython.Tools
import image.StructuralAPI

def createScene(rootNode):
    rootNode.createObject("RequiredPlugin", name="image")
    im = image.StructuralAPI.Image(rootNode,"armadillo")
    im.addMeshLoader("mesh/Armadillo_simplified.obj")
    im.addImage(0.5, 1)
    im.addViewer()
    rootNode.createObject("OglModel", name="visu",  src="@"+SofaPython.Tools.getObjectPath(im.mesh), color="0.5 0.5 1 .5")