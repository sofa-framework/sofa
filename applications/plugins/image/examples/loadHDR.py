def createScene(node):
    node.createObject('RequiredPlugin', name="image")
    node.createObject('BackgroundSetting',color='1 1 1')
    node.createObject('ImageContainer', template="ImageUS", name="image", filename="data/knee.hdr")
    node.createObject('ImageViewer', template="ImageUS", name="viewer", src="@image", plane="70 70 70")

