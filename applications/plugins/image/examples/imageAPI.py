import SofaPython.Tools
import SofaImage.API

def createScene(rootNode):
    rootNode.createObject("RequiredPlugin", name="Image")
    im = SofaImage.API.Image(rootNode,"armadillo")
    im.addMeshLoader("mesh/Armadillo_simplified.obj",1,name="armadillo")
    im.addMeshToImage(0.5)
    im.addViewer()
    im.addExporter()
    im.addMeshVisual("armadillo")
    im.meshes["armadillo"].visual.setColor(.5, 0.5, 1, .5)
    
    im2 = SofaImage.API.Image(rootNode,"armadillo2")
    im2.addContainer(im.getFilename())
    im2.addViewer()