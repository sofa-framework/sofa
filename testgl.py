def createScene(root):
        for i in range(10):
                root.createObject("OglModel", name="armadille"+str(i), filename="mesh/Armadillo_simplified.obj", forceFloat=True)
