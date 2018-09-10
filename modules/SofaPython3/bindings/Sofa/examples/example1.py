print(".EXAMPLE SCENE.")
import SofaPython3

def createScene(rootNode):
        print("HELLO WORLD: "+str(type(rootNode)))
        print("Dir:"+str(dir(rootNode)))
        
        rootNode.createChild("Child1")
        rootNode.createChild("Child2")
        
        return rootNode
