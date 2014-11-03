import Sofa

import sys


def createScene(root):
    
    # root node setup
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
    
    # simuation parameters
    root.dt = 1e-2
    root.gravity = [0, -10, 0]
    
    
    # scene nodes
    inside = root.createChild('insideScope')
    outside = root.createChild('outsideScope')
    common = inside.createChild('common')
    extension = common.createChild('extension')
    
    ode = inside.createObject('CompliantImplicitSolver')
    ode.stabilization = "pre-stabilization"
    num = inside.createObject('LDLTSolver')
    insidedof = inside.createObject('MechanicalObject',name='insidedof', position="1 0 0")
    inside.createObject('UniformMass',name='mass',totalmass='1')
    
    
    outsidedof = outside.createObject('MechanicalObject',name='outsidedof', position="0 0 0")
    outside.createObject('UniformMass',name='mass',totalmass='1')
    
    commondof = common.createObject('MechanicalObject',name='commondof')    
    common.createObject('SubsetMultiMapping', template='Vec3d,Vec3d', input='@../insidedof @../../outsideScope/outsidedof', output='@commondof' , indexPairs="1 0  0 0" )	
    outside.addChild( common )
    
    
    extensiondof = extension.createObject('MechanicalObject',template="Vec1d",name='extensiondof')
    extension.createObject('EdgeSetTopologyContainer',edges="0 1")
    extension.createObject('UniformCompliance',template="Vec1d",compliance="0")
    extension.createObject('DistanceMapping',template="Vec3d,Vec1d")
