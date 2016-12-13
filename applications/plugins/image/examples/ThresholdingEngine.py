import Sofa

def createScene(node):
    node.createObject('RequiredPlugin',name="image")

    node.createObject('ImageContainer', template="ImageUC", name="img", filename="data/depth0014-scale.pgm", drawBB="1")


    node.createObject('ThresholdingEngine', template="ImageUC", name="engine", method="1", param="500", src="@img" )

    node.createObject('ImageFilter', template="ImageUC,ImageUC", name="filter", filter="13", param="@engine.threshold",  src="@img" )

    node.createObject('ImageViewer', template="ImageUC",  name="viewer", src="@filter")
    # node.createObject('ImageViewer', template="ImageUC",  name="viewer2", src="@img")
