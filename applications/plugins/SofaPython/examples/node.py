import Sofa 

def createScene(rootNode):
    A = rootNode.createChild('A')
    A.createObject("MechanicalObject", name="dofs")
    print A.getObject("dofs").name,":",A.getObject("dofs").position