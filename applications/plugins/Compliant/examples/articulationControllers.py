import Sofa

from Compliant import StructuralAPI, Tools

path = Tools.path( __file__ )

import sys


# keep an eyes on the controllers
#hingebasearm0_positionController = None
#hingearm0arm_velocityController = None
#sliderfingerLarm_forceController = None
#sliderfingerRarm_forceController = None

#limits
hingebasearm0_limit = 0.5

# control values
hingebasearm0_offset = 0
hingearm0arm_velocity = 0
sliderfinger_force = 10000

    
def createScene(root):
    
    
    
    #### print HELP
    print "\n\n################"
    print "'UP' and 'DOWN' to control hinge position between base and arm0"
    print "'LEFT' and 'RIGHT' to control hinge velocity between arm0 and arm"
    print "'+' and '-' to control finger slider forces"
    print "(do no forget to press CTRL)"
    print "################\n"
    sys.stdout.flush()
    
    
    
    
  
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehaviorModels" )
    root.dt = 0.001
    root.gravity = [0, -9.8, 0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('CompliantAttachButtonSetting')
    
    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization=1)
    root.createObject('SequentialSolver', iterations=100)
    root.createObject('LDLTResponse')
    
    
    
    
    ### OBJETS
    
    
    
    base = StructuralAPI.RigidBody( root, "base" )
    base.setFromMesh( "mesh/PokeCube.obj", 1000, [0,-4,0,0,0,0,1], [1,10,1] )
    cm = base.addCollisionMesh( "mesh/PokeCube.obj", [1,10,1] )
    cm.triangles.group = "1"
    cm.addVisualModel()
    base.node.createObject('FixedConstraint')
    
    basearmoffset = base.addOffset( "basearmoffset", [0,4.5,0,0,0,0,1] )
    basearmoffset.dofs.showObject=True
    basearmoffset.dofs.showObjectScale=1
    
    
    
    arm0 = StructuralAPI.RigidBody( root, "arm0" )
    arm0.setFromMesh( "mesh/PokeCube.obj", 100, [0,5,0,0,0,0,1], [.9,10,.9] )
    cm = arm0.addCollisionMesh( "mesh/PokeCube.obj", [.9,10,.9] )
    cm.triangles.group = "1"
    vm = cm.addVisualModel()
    vm.model.setColor( 1,1,0,1 )
    
    armbaseoffset = arm0.addOffset( "armbaseoffset", [0,-4.5,0,0,0,0,1] )
    armbaseoffset.dofs.showObject=True
    armbaseoffset.dofs.showObjectScale=1
    
    arm0armoffset = arm0.addOffset( "arm0armoffset", [0,4.5,0,0,0,0,1] )
    arm0armoffset.dofs.showObject=True
    arm0armoffset.dofs.showObjectScale=1
    
    
    
    
    arm = StructuralAPI.RigidBody( root, "arm" )
    arm.setFromMesh( "mesh/PokeCube.obj", 10, [0,14,0,0,0,0,1], [.8,10,.8] )
    cm = arm.addCollisionMesh( "mesh/PokeCube.obj", [.8,10,.8] )
    cm.triangles.group = "1"
    vm = cm.addVisualModel()
    vm.model.setColor( 1,0,0,1 )
    
    armarm0offset = arm.addOffset( "armarm0offset", [0,-4.5,0,0,0,0,1] )
    armarm0offset.dofs.showObject=True
    armarm0offset.dofs.showObjectScale=1
    
    armfingeroffset = arm.addOffset( "armfingeroffset", [0,4.75,0,0,0,0,1] )
    armfingeroffset.dofs.showObject=True
    armfingeroffset.dofs.showObjectScale=1
    
    
    
    
    
    fingerL = StructuralAPI.RigidBody( root, "fingerL" )
    fingerL.setFromMesh( "mesh/PokeCube.obj", 1, [-1,20,0,0,0,0,1], [.5,3,.5] )
    cm = fingerL.addCollisionMesh( "mesh/PokeCube.obj", [.5,3,.5] )
    cm.triangles.group = "1"
    vm = cm.addVisualModel()
    vm.model.setColor( 0,1,0,1 )
    
    fingerLarmoffset = fingerL.addOffset( "fingerLarmoffset", [0,-1.25,0,0,0,0,1] )
    fingerLarmoffset.dofs.showObject=True
    fingerLarmoffset.dofs.showObjectScale=1
    
    
    
    fingerR = StructuralAPI.RigidBody( root, "fingerR" )
    fingerR.setFromMesh( "mesh/PokeCube.obj", 1, [1,20,0,0,0,0,1], [.5,3,.5] )
    cm = fingerR.addCollisionMesh( "mesh/PokeCube.obj", [.5,3,.5] )
    cm.triangles.group = "1"
    vm = cm.addVisualModel()
    vm.model.setColor( 0,0,1,1 )
    
    
    fingerRarmoffset = fingerR.addOffset( "fingerRarmoffset", [0,-1.25,0,0,0,0,1] )
    fingerRarmoffset.dofs.showObject=True
    fingerRarmoffset.dofs.showObjectScale=1
    
    
    
    ##### LINKS
    
    global hingebasearm0_positionController
    global hingearm0arm_velocityController
    global sliderfingerLarm_forceController
    global sliderfingerRarm_forceController
    
    hingebasearm = StructuralAPI.HingeRigidJoint( 2, "hingebasearm", basearmoffset.node, armbaseoffset.node )
    
    hingebasearm0_positionController = hingebasearm.addPositionController(hingebasearm0_offset,1e-8)
    hingebasearm.addLimits(-hingebasearm0_limit,hingebasearm0_limit)
    
    
    hingearm0arm = StructuralAPI.HingeRigidJoint( 2, "hingearm0arm", arm0armoffset.node, armarm0offset.node )
    hingearm0arm_velocityController = hingearm0arm.addVelocityController(hingearm0arm_velocity,0)
    #hingearm0arm.addLimits(-.75,.75)  # limits with velocity controller are not well handled
    
    
    
    sliderfingerLarm = StructuralAPI.SliderRigidJoint( 0, "sliderfingerLarm", armfingeroffset.node, fingerLarmoffset.node )
    sliderfingerLarm_forceController = sliderfingerLarm.addForceController(-sliderfinger_force)
    sliderfingerLarm.addLimits(-2,-0.25)
    
    
    
    
    sliderfingerRarm = StructuralAPI.SliderRigidJoint( 0, "sliderfingerRarm", armfingeroffset.node, fingerRarmoffset.node )
    sliderfingerRarm_forceController = sliderfingerRarm.addForceController(sliderfinger_force)
    sliderfingerRarm.addLimits(0.25,2)
    
    
    
    root.createObject('PythonScriptController', filename = __file__, classname = 'Controller')





class Controller(Sofa.PythonScriptController):
     
    def onLoaded(self,node):
        return 0
          
    
    def onBeginAnimationStep(self, dt):
        return 0
    
    # key and mouse events; use this to add some user interaction to your scripts 
    def onKeyPressed(self,k):
        
        global hingebasearm0_positionController
        global hingearm0arm_velocityController
        global sliderfingerLarm_forceController
        global sliderfingerRarm_forceController
        
        global hingebasearm0_offset
        global hingearm0arm_velocity
        global sliderfinger_force
            
        # UP key -> hinge angle between base and arm0
        if ord(k)==19:
            hingebasearm0_offset += 0.1
            if( hingebasearm0_offset > hingebasearm0_limit ) :
                hingebasearm0_offset = hingebasearm0_limit
            hingebasearm0_positionController.setOffsets([hingebasearm0_offset])
            print "Controlling hinge 'base-arm0' position "+str(hingebasearm0_offset)
            
        # DOWN key -> hinge angle between base and arm0
        elif ord(k)==21:
            hingebasearm0_offset -= 0.1
            if( hingebasearm0_offset < -hingebasearm0_limit ) :
                hingebasearm0_offset = -hingebasearm0_limit
            hingebasearm0_positionController.setOffsets([hingebasearm0_offset])
            print "Controlling hinge 'base-arm0' position "+str(hingebasearm0_offset)
            
        # RIGHT key -> hinge velocity between arm and arm0
        elif ord(k)==20:
            hingearm0arm_velocity -= 0.2
            if( hingebasearm0_offset < -1 ) :
                hingebasearm0_offset = -1
            hingearm0arm_velocityController.setVelocities([hingearm0arm_velocity])
            print "Controlling hinge 'arm0-arm' velocity "+str(hingearm0arm_velocity)
            
        #LEFT key -> hinge velocity between arm and arm0
        elif ord(k)==18:
            hingearm0arm_velocity += 0.2
            if( hingebasearm0_offset > 1 ) :
                hingebasearm0_offset = 1
            hingearm0arm_velocityController.setVelocities([hingearm0arm_velocity])
            print "Controlling hinge 'arm0-arm' velocity "+str(hingearm0arm_velocity)
            
        # + -> force between fingers and arm
        elif k=='+':
            sliderfinger_force = 1000000
            sliderfingerLarm_forceController.setForces([-sliderfinger_force])
            sliderfingerRarm_forceController.setForces([sliderfinger_force])
            print "Controlling finger slider forces "+str(sliderfinger_force)
            
        # - -> force between fingers and arm
        elif k=='-':
            sliderfinger_force = -1000000
            sliderfingerLarm_forceController.setForces([-sliderfinger_force])
            sliderfingerRarm_forceController.setForces([sliderfinger_force])
            print "Controlling finger slider forces "+str(sliderfinger_force)
        else:
            print "Wrong key: " + k + " " + str(ord(k))
        
        sys.stdout.flush()
        return 0 
    