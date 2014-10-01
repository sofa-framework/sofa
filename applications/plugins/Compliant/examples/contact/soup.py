import Sofa


FRICTION = 1


def createScene(root):
  
  	########################    root node setup    ########################
	root.createObject('RequiredPlugin', pluginName = 'Compliant')
	root.createObject('VisualStyle', displayFlags = "hideBehavior showCollisionModels" )
	
	
	########################    simulation parameters    ########################
      
	root.findData('dt').value=0.02
	root.findData('gravity').value=[0,0,-10]

	
	########################    global components    ########################
	
	root.createObject('CompliantAttachButtonSetting')
	root.createObject('DefaultPipeline', name='DefaultCollisionPipeline', depth="6")
	root.createObject('BruteForceDetection')
	#root.createObject('IncrementalSweepAndPrune')
	root.createObject('NewProximityIntersection', name="Proximity", alarmDistance="0.2", contactDistance="0")
	root.createObject('DefaultCollisionGroupManager')
	
	if FRICTION:
		root.createObject('DefaultContactManager', name="Response", response="FrictionCompliantContact", responseParams="compliance=0&restitution=0&mu=0.1" )
	else:
		root.createObject('DefaultContactManager', name="Response", response="CompliantContact", responseParams="compliance=0&restitution=0" )
	
	
	root.createObject('CompliantImplicitSolver',stabilization="1")
	root.createObject('SequentialSolver',iterations="100",precision="1e-15")
		
	
	
	########################    fixed collider    ########################
	
	floorNode = root.createChild('Floor')
	floorNode.createObject('MechanicalObject',template="Rigid",position="0 0 0 0 0 0 1")
	floorNode.createObject('FixedConstraint',template="Rigid",indices="0")
	
	floorColNode = floorNode.createChild('FloorCol')	
	floorColNode.createObject('MeshObjLoader', name='loader', filename='mesh/SaladBowl.obj',scale3d="50 50 50")
	floorColNode.createObject('Mesh', src='@loader')
	floorColNode.createObject('MechanicalObject')
	floorColNode.createObject('RigidMapping')
	
	floorColNode.createObject('Triangle', name='Floor', simulated=0, moving=0, proximity=0.01)

	
	########################    spheres   ########################
	
	
	####### SELF-COLLISION
	
	spherePos = ""
	for x in range(-3,3):
		for y in range(-3,3):
			spherePos += str(x*2.5)+" "+str(y*2.5)+" 20  "
	
	sphereNode = root.createChild('Spheres')
	sphereNode.createObject('MechanicalObject',template='Vec3d',position=spherePos, velocity='0 0 -1')
	sphereNode.createObject('TSphereModel',template='Vec3d',name='sphere_model',radius=1,selfCollision="1")
	sphereNode.createObject('UniformMass',name='mass',mass=.1)
	
	
	####### INTER-COLLISION
	
	i=0
	for x in range(-3,3):
		for y in range(-3,3):
                        for z in xrange(1):
                            sphereNode = root.createChild('Sphere'+str(i))
                            sphereNode.createObject('MechanicalObject',template='Vec3d',position=str(x*2.5)+" "+str(y*2.5)+" "+str(20+2.5*(z+1)), velocity='0 0 -1')
                            sphereNode.createObject('TSphereModel',template='Vec3d',name='sphere_model',radius=1,selfCollision="0")
                            sphereNode.createObject('UniformMass',name='mass',mass=.1)
                            i+=1
		
		

	return 0




