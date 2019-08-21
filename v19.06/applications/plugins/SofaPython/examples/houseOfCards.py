#!/usr/bin/python
import math
import Sofa



#### user defines

sizeHouseOfCards=4
friction=math.sqrt(0.8)
contactDistance=0.03
    
    
    
    
#####
    
angle=20.0
distanceInBetween=0.1




def convertDegreeToRadian(a):
	return a*3.14159265/180.0



def createCard(node,position,rotation):
	
	cardNode = node.createChild('Card')
	
	cardNode.createObject('EulerImplicit',name='cg_odesolver',printLog='false',rayleighStiffness='0.1', rayleighMass='0.1')
	cardNode.createObject('CGLinearSolver',name='linear solver',iterations='15',tolerance='1.0e-5',threshold='1.0e-5')
	cardNode.createObject('LMConstraintSolver', listening="1",  constraintVel="1",  constraintPos="1", numIterations=25)
	
	posStr = str(position[0])+' '+str(position[1])+' '+str(position[2])
	rotStr = str(rotation[0])+' '+str(rotation[1])+' '+str(rotation[2])

	cardNode.createObject('MechanicalObject',name='Rigid Object',template='Rigid',translation=posStr,rotation=rotStr)
        cardNode.createObject('UniformMass',totalmas=0.5,filenameMass="BehaviorModels/card.rigid")

        # VisualNode
        VisuNode = cardNode.createChild('Visu')
        VisuNode.createObject('OglModel',name='Visual',fileMesh='mesh/card.obj')
        VisuNode.createObject('RigidMapping',input='@..',output='@.')

        # VisualNode
        collNode = cardNode.createChild('collision')
        collNode.createObject('MeshObjLoader', name='loader', filename='mesh/cardCollision.obj')
	collNode.createObject('Mesh', src='@loader')
	collNode.createObject('MechanicalObject', src='@loader')
	collNode.createObject('Triangle', name='cardt', contactFriction=friction)
	collNode.createObject('Line', name='cardl',  contactFriction=friction)
	collNode.createObject('Point', name='cardp', contactFriction=friction)
        collNode.createObject('RigidMapping',input='@..',output='@.')

	return cardNode;
	
	
	
def create2Cards(node,globalPosition):
	#We assume the card has a 2 unit length, centered in (0,0,0)
	displacement=math.sin(convertDegreeToRadian(angle));
	separation=displacement+distanceInBetween;

	#Left Rigid Card
	positionLeft=[globalPosition[0]+separation,globalPosition[1],globalPosition[2]]
	rotationLeft=[0,0,angle]
	leftCard = createCard(node, positionLeft, rotationLeft)

	#Right Rigid Card
	positionRight=[globalPosition[0]-separation,globalPosition[1],globalPosition[2]]
	rotationRight=[0,0,-angle]
	rightCard = createCard(node, positionRight, rotationRight)

	return 0

	
	
def createHouseOfCards(node):
	#Space between two levels of the house of cards
	space=0.75
	#Size of a card
	sizeCard=2
	#overlap of the cards
	factor=0.95
	distanceH=sizeCard*factor;
	distanceV=sizeCard*math.cos(convertDegreeToRadian(angle))+space;
	
	for i in xrange(0,sizeHouseOfCards):
		# 2 cards
		for j in xrange(0,i+1):
			position=[(i+j)*(distanceH)*0.5,(i-j)*(distanceV),0]
			create2Cards(node,position)
			
		
		# support for the cards
		initPosition=sizeCard*0.5
		supportRotation=[0,0,90]
		for j in xrange(0,i):
			position=[(i+j)*(distanceH)*0.5,(i-j)*distanceV-space*0.5+distanceInBetween*(j%2)-initPosition,0]
			createCard(node, position, supportRotation);

			
	return 0
    
    

# scene creation method
def createScene(rootNode):

	rootNode.findData('dt').value = 0.001
	rootNode.findData('gravity').value=[0.0, -10, 0.0]

	rootNode.createObject('CollisionPipeline', verbose=0, depth=10, draw=0)
	rootNode.createObject('BruteForceDetection', name='N2')
	rootNode.createObject('MinProximityIntersection', name='Proximity', alarmDistance=contactDistance, contactDistance=contactDistance*0.5)
	rootNode.createObject('DefaultContactManager', name='Response', response='distanceLMConstraint')
	rootNode.createObject('DefaultCollisionGroupManager')
	
	#rootNode.createObject('EulerImplicit',name='cg_odesolver',printLog='false',rayleighStiffness='0.1', rayleighMass='0.1')
	#rootNode.createObject('CGLinearSolver',name='linear solver',iterations='15',tolerance='1.0e-5',threshold='1.0e-5')
	#rootNode.createObject('LMConstraintSolver', listening="1",  constraintVel="1",  constraintPos="1", numIterations=25)
				
	# floor
	floorNode = rootNode.createChild('Floor')
	floorNode.createObject('MeshObjLoader', name='loader', filename='mesh/floor.obj')
	floorNode.createObject('Mesh', src='@loader')
	floorNode.createObject('MechanicalObject', src='@loader')
	floorNode.createObject('Triangle', name='Floor', simulated=0, moving=0, contactFriction=friction)
	floorNode.createObject('Line', name='Floor', simulated=0, moving=0, contactFriction=friction)
	floorNode.createObject('Point', name='Floor', simulated=0, moving=0, contactFriction=friction)
	floorNode.createObject('OglModel', name='FloorV', filename='mesh/floor.obj')
	
	
	# castle
	createHouseOfCards( rootNode );
  
  
	return rootNode