import Sofa
import SofaPython.DAGValidation


def createScene( node ):
    
    node.createObject('MechanicalObject', name="dof_0", template='Vec3d', position="0 0 0")
    
    child00 = node.createChild('child00')
    child00.createObject('MechanicalObject', name="dof_00", template='Vec3d')
    child00.createObject('IdentityMapping')
    
    
    child01 = node.createChild('child01')
    child01.createObject('MechanicalObject', name="dof_01", template='Vec3d')
    child01.createObject('IdentityMapping')
    
    
    
    child000 = child00.createChild('child000')
    child000.createObject('MechanicalObject', name="dof_000", template='Vec3d')
    child000.createObject('SubsetMultiMapping', template='Vec3d,Vec3d', input='@../../child01/ @../', output='@/', indexPairs="0 0 1 0" )

#####################"
    
    print "$$$ should be invalid here (missing parent)"
    SofaPython.DAGValidation.test( node )
    
#####################"

    child01.addChild( child000 )
    
    print "$$$ should be valid here"
    SofaPython.DAGValidation.test( node )
    
#####################"    
    
    child02 = child000.createChild('child02')
    child02.createObject('MechanicalObject', name="dof_02", template='Vec3d')
    
    print "$$$ should be invalid here (independent dof under other dofs)"
    SofaPython.DAGValidation.test( node )
    
    child000.removeChild( child02 )
    
#####################"
    
    node.addChild( child000 )
    
    print "$$$ should be invalid here (too many parents)"
    SofaPython.DAGValidation.test( node )
    
    