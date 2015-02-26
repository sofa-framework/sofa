import Sofa
import SofaTest
import math

from Compliant import StructuralAPI, Tools

path = Tools.path( __file__ )

# This sample will bug with a non zero sphereInitialSpeed and animateRopeLength set.
# It is stable however if you don't animate the rope length, or don't give an initial speed to the sphere.

# shared data
class Shared:
    pass

global shared
shared = Shared()


class DistanceJoint:

    def __init__(self, name, node1, node2, compliance=0, index1=0, index2=0, rest_lenght=-1 ):
        self.node = node1.createChild( name )
        self.dofs = self.node.createObject('MechanicalObject', template='Vec3d', name='dofs' )
        input = '@' + Tools.node_path_abs(node1) + '/dofs' + ' @' + Tools.node_path_abs(node2) + '/dofs'
        self.mapping = self.node.createObject('SubsetMultiMapping', template='Vec3d,Vec3d', name='mapping', input = input, output = '@dofs', indexPairs="0 "+str(index1)+" 1 "+str(index2) )
        self.constraint = DistanceJoint.Constraint(self.node, compliance, rest_lenght)
        node2.addChild( self.node )


    class Constraint:
        def __init__(self, node, compliance, rest_length ):
            self.node = node.createChild( 'constraint' )
            self.dofs = self.node.createObject('MechanicalObject', template = 'Vec1d', name = 'dofs', position = '0' )
            self.topology = self.node.createObject('EdgeSetTopologyContainer', edges="0 1" )
            self.mapping = self.node.createObject('DistanceMapping',  name='mapping', rest_length=(rest_length if rest_length>0 else "" ) )
            self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
            #self.stabilization = self.node.createObject('Stabilization')

class Rope:
    def __init__(self,parentNode,name,x,y,z,endx,endy,endz,segments,mass,compliance):
        self.node = parentNode.createChild(name)
        positions=''
        velocities=''
        lengths=''
        extvelocities=''
        edges=''
        dx = endx - x
        dy = endy - y
        dz = endz - z
        seglen = math.sqrt(dx*dx+dy*dy+dz*dz) / float(segments)
        for i in range(0, segments+1):
            f = float(i) / float(segments)
            positions += str(x + f * dx)+' '+str(y + f * dy)+' '+str(z + f * dz)+' '
            velocities += '0 0 0 '
            if i < segments:
                lengths += str(seglen) + ' '
                extvelocities += '0 '
                edges += str(i) + ' ' + str(i+1) + ' '
        
        self.dofs = self.node.createObject('MechanicalObject', template='Vec3d', name='dofs', position=positions, velocity=velocities)
        self.mass = self.node.createObject('UniformMass',name='mass',totalmass=str(mass))
        self.extensionNode = self.node.createChild('extension')
        self.edgesDofs = self.extensionNode.createObject('MechanicalObject', template='Vec1d', name='dofs', position=extvelocities, velocity=extvelocities)
        self.edges = self.extensionNode.createObject('EdgeSetTopologyContainer', edges=edges)
        self.distanceMapping = self.extensionNode.createObject('DistanceMapping',  name='mapping', restLengths=lengths)
        self.extensionNode.createObject('UniformCompliance', name='compliance', compliance=compliance)
        #self.stabilization = self.extensionNode.createObject('Stabilization')


def createFixedParticle(parentNode,name,x,y,z):
    node = parentNode.createChild(name)
    node.createObject('MechanicalObject', template='Vec3d', name='dofs', position=str(x)+' '+str(y)+' '+str(z), velocity='0 0 0')
    node.createObject('UniformMass',name='mass',totalMass='1')
    node.createObject('FixedConstraint')
    return node

def createRigidSphere(parentNode,name,x,y,z,vx,vy,vz,radius,mass):
    node = parentNode.createChild(name)
    node.createObject('MechanicalObject', template='Rigid3d', name='dofs', position=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1', velocity=str(vx)+' '+str(vy)+' '+str(vz)+' 0 0 0 1')
    node.createObject('TSphereModel',template='Rigid3d',name='sphere_model',radius=str(radius))
    node.createObject('UniformMass',name='mass',totalMass=mass)
    return node

def createRigidMappedParticle(parentNode,name,x,y,z):
    node = parentNode.createChild(name)
    node.createObject('MechanicalObject', template='Vec3d', name='dofs', position=str(x)+' '+str(y)+' '+str(z), velocity='0 0 0')
    node.createObject('RigidMapping',name='mapping', input = '@' + Tools.node_path_rel(node,parentNode) + '/dofs' , output = '@dofs')
    return node

def createScene(root):
    
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showAll" )
    root.dt = 1.0/120.0
    root.gravity = [0, -9.8, 0]
    
    root.createObject('RequiredPlugin', name='Compliant', pluginName = 'Compliant')
    #root.createObject('CompliantAttachButtonSetting')
    
    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization=1, warm_start=1, stabilization_damping=0, neglecting_compliance_forces_in_geometric_stiffness=0)
    root.createObject('SequentialSolver', iterations=100, precision=1e-10, relative=0)
    root.createObject('LDLTResponse')
    #root.createObject('CompliantSleepController', name="Sleep Controller", listening=1, immobileThreshold=0.0001, rotationThreshold=0.01)
    ##### OBJECTS
    
    ropeLength = 10
    ropeSegments = 10
    ropeMass = 10
    sphereRadius = 1
    sphereMass = 4
    ropeCompliance = 1e-10
    attachCompliance = 1e-5
    sphereInitialSpeed = 10.0
    
    baseNode = root.createChild('base')
    bg = createFixedParticle(baseNode, "bg", 0, ropeLength, 0)
    weight = createRigidSphere(baseNode, "weight", 0, -sphereRadius, 0, sphereInitialSpeed, 0, 0, sphereRadius, sphereMass)
    weightReference = createRigidMappedParticle(weight, "particle", 0, 0, 0)
    rope = Rope(baseNode, "rope", 0, ropeLength, 0, 0, 0, 0, ropeSegments, ropeMass, ropeCompliance)
    ropeAttachBg = DistanceJoint("c_rope_bg", rope.node, bg, attachCompliance, 0, 0)
    ropeAttachWeight = DistanceJoint("c_rope_weight", rope.node, weightReference, attachCompliance, ropeSegments, 0)
    
    script = root.createObject('PythonScriptController', filename = __file__, classname = 'Controller')
    
    shared.distanceMapping = rope.distanceMapping
    shared.testMO = weight.getObject('dofs')


class Controller(SofaTest.Controller):
    
    iterations = 0
    ropeDelta = 0
    changeSpeed = 1.0 / 600.0
    animateRopeLength = 1
    
    def onEndAnimationStep(self, dt):
        self.iterations = self.iterations + 1
        
        if self.iterations == 1:
            self.numRopeSegments = len(shared.distanceMapping.restLengths)
            self.ropeLength = shared.distanceMapping.restLengths[0][0]
            self.minRopeLength = self.ropeLength * 0.3 
            self.maxRopeLength = self.ropeLength
            self.ropeDelta = - self.ropeLength * self.changeSpeed
        
        if self.animateRopeLength:
            self.ropeLength += self.ropeDelta
            if self.ropeLength < self.minRopeLength:
                self.ropeDelta = -self.ropeDelta
                self.ropeLength = self.minRopeLength
            if self.ropeLength > self.maxRopeLength:
                self.ropeDelta = -self.ropeDelta
                self.ropeLength = self.maxRopeLength
                
            newValue = ''
            for i in range(0, self.numRopeSegments):
                newValue += str(self.ropeLength) + ' '
            
            #print "iteration " + str(self.iterations) + " len = " + str(self.ropeLength * self.numRopeSegments)
            #print "iteration " + str(self.iterations) + " pos: " + str(shared.testMO.position[0])
            shared.distanceMapping.restLengths = newValue
        
        if self.iterations >= 1000:
            print "Attached object end position : " + str(shared.testMO.position[0])
            posx = shared.testMO.position[0][0]
            self.should( posx > -11 and posx < 11, 'object became unstable' )
