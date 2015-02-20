import SofaPython.Tools
from SofaImage import ImageAPI

def createScene(rootNode):
    rootNode.createObject("RequiredPlugin", name="Image")
    im = ImageAPI.Image(rootNode,"armadillo")
    im.addMeshLoader("mesh/Armadillo_simplified.obj")
    im.addImage(0.5, 1)
    im.addViewer()
    im.addExporter("armadillo.mhd")
    rootNode.createObject("OglModel", name="visu",  src="@"+SofaPython.Tools.getObjectPath(im.mesh), color="0.5 0.5 1 .5")
    
    im2 = ImageAPI.Image(rootNode,"armadillo2")
    im2.addContainer("armadillo.mhd")
    im2.addViewer()