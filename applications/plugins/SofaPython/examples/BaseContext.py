import Sofa
import sys


def printObjects( title, objects ):
    print "***",title,":"
    for o in objects:
        print "   ",o.getPathName(), type(o)



def createScene(node):

    node.createObject('MechanicalObject')
    node.createObject('UniformMass')
    childNode = node.createChild("child")
    childNode.createObject('MechanicalObject')
    childNode.createObject('IdentityMapping',name="toto")
    childNode.createObject('DiagonalMass')
    grandchildNode = childNode.createChild("grandchild")
    grandchildNode.createObject('MechanicalObject',name="toto")
    grandchildNode.createObject('IdentityMapping')


    # getObjects

    printObjects( "child / Local / all", childNode.getObjects() )


    # getObjects by type

    printObjects( "grandchild / SearchRoot / all", grandchildNode.getObjects("SearchRoot") )
    printObjects( "child / SearchDown / MechanicalState", childNode.getObjects("SearchDown","MechanicalState") )
    printObjects( "grandchild / SearchUp / Mass", grandchildNode.getObjects("SearchUp","Mass") )
    printObjects( "grandchild / SearchUp / UniformMass", grandchildNode.getObjects("SearchUp","UniformMass") )


    # getObjects by name

    printObjects( "root / SearchRoot / BaseObject / toto", node.getObjects("SearchRoot","BaseObject","toto") )


    # getObjects by type and name

    printObjects( "root / SearchRoot / Mapping / toto", node.getObjects("SearchRoot","Mapping","toto") )



    sys.stdout.flush()