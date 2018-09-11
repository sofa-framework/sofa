print(".EXAMPLE SCENE.")

def createScene(rootNode):
        print("COUCOU")
        print("Type: "+str(type(rootNode)))
        print("Name: "+str(type(rootNode.getName())))
        
        one = rootNode.createChild("Child1")
        two = rootNode.createChild("Child2")
        
        print("T:" + str(one.getName()))
        print("T:" + str(two.getName()))
        
        return rootNode
