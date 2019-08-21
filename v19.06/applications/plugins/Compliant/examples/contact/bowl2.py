import Sofa

import math

from Compliant import StructuralAPI, Tools
from Compliant.types import Quaternion, Rigid3
import numpy as np

dir = Tools.path( __file__ )

def createScene(node):
    scene = Tools.scene( node )

    style = node.getObject('style')
    style.findData('displayFlags').showMappings = False
    style.findData('displayFlags').showVisual = False
    style.findData('displayFlags').showCollision = True    

    manager = node.getObject('manager')
    manager.response = 'PenalityCompliantContact'

    manager.responseParams="stiffness=1e5"

    node.dt = 2.5e-3
    # node.createObject('CompliantAttachButton')
                            
    ode = node.getObject('ode')
    node.removeObject(ode)
    ode = node.createObject("EulerImplicitSolver", rayleighStiffness = 0, rayleighMass = 0)
    
    num = node.createObject('CGLinearSolver',
                            name = 'num',
                            iterations = 100,
                            tolerance = 1e-7, threshold = 0)
    num.printLog = 1
    # num = node.createObject('CgSolver', name = 'num', iterations = 100, precision = 1e-7 )
    
    proximity = node.getObject('proximity')

    proximity.alarmDistance = 0.1
    proximity.contactDistance = 0
    proximity.useLineLine = True

  
    # planes
    for i in xrange(4):
        mesh = dir + '/../mesh/ground.obj'
        plane = StructuralAPI.RigidBody( node, "plane-{}".format(i) )

        g = Rigid3()

        n = np.zeros(3)
        n[:] = [0, 1, 0]

        r = Quaternion.exp( math.pi / 8 * np.array([1.0, 0.0, 0.0]))
        
        q = Quaternion.exp(i * math.pi / 2 * np.array([0.0, 1.0, 0.0]))

        g.orient = q * r
        
        plane.setManually( g, 1, [1,1,1] )
        
        #body3.setFromMesh( mesh, 1 )
        cm = plane.addCollisionMesh( mesh, [10, 1, 10] )
        cm.triangles.group = "0"
        
        cm.addVisualModel()
        plane.node.createObject('FixedConstraint', indices = '0')
        cm.triangles.contactFriction = 0.5 # per object friction coefficient
        
    # box

    particles = node.createChild('particles')
    n = 400


    for i in xrange(n):
        p = particles.createChild('particle-{0}'.format(i))
        dofs = p.createObject('MechanicalObject', template = "Vec3d")

        pos = np.zeros( 3 )
        vel = np.zeros( 3 )    

        pos[1] = 2 + i / 1.5
        vel[1] = -1        

        dofs.position = pos.tolist()
        dofs.velocity = vel.tolist()
    
        mass = p.createObject('UniformMass', template = 'Vec3d')
        model = p.createObject('SphereModel', template = 'Vec3d',
                               selfCollision = True,
                               radius = 0.1)

    # mesh = dir + '/../mesh/cube.obj'
    # box = StructuralAPI.RigidBody( node, "box" )
    # box.setFromMesh( mesh, 50, [0, 3, 0, 0,0,0,1] )
    # cm = box.addCollisionMesh( mesh )
    # cm.addVisualModel()
    # cm.triangles.contactFriction = 1 # per object friction coefficient
    # cm.triangles.group = "1"
    
