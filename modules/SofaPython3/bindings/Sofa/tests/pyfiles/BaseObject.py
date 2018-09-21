import Sofa

def OnEven(self):
        yield (0,0)
        yield (1,1)

def createScene(rootNode):
        node = rootNode.createChild("root")
        obj = node.createObject("MechanicalObject", name="mmm", position=[[1.0,2.0,3.0],[4.0,5.0,6.0]])
        print(str(type(obj.name)))
        print(str(type(obj.position)))
        
        print(str(len(obj.name)))
        print(str(len(obj.position)))
        
        print(":"+str(obj.position.dim()))

        print("position[1]: "+str(obj.position[0]))
        print("position[0:1]: "+str(obj.position[0:1]))
        print("position[1,1]: "+str(obj.position[1,1]))
        print("position[0:1,0:]: "+str(obj.position[0:1,0:]))
        print("position[0:1,0:]: "+str(obj.position[OnEven]))
        

        #ASSERT_NEQ( node.getObject("m"), None )
        #ASSERT_NEQ( node.m.position, None )
        #ASSERT_EQ( node.m.position.tolist(), [[1.0,2.0,3.0],[4.0,5.0,6.0]] )
