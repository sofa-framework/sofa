print(".EXAMPLE SCENE.")

def createScene(rootNode):
        print("COUCOU")
        print("HELLO WORLD: "+str(type(rootNode)))
        print("Dir:"+str(dir(rootNode)))
        
        rootNode.createChild("Child1")
        rootNode.createChild("Child2")
        
        return rootNode
