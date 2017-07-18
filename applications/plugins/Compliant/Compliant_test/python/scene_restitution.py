import Sofa, SofaTest


coefs = [0,0.1,0.25,0.5,0.75,0.9,1]
vel = -10
        
def createScene(root):
  
        ########################    root node setup    ########################
        root.createObject('RequiredPlugin', pluginName = 'Compliant')
        root.createObject('VisualStyle', displayFlags = "hideBehavior showCollisionModels" )
        
        ########################    python test controller    ########################
        
        root.createObject('PythonScriptController', filename = __file__, classname = 'TestController')
        
        ########################    simulation parameters    ########################
      
        root.findData('dt').value=0.01
        root.findData('gravity').value=[0,0,0]

        
        ########################    global components    ########################
        
        root.createObject('DefaultPipeline', name='DefaultCollisionPipeline', depth="6")
        root.createObject('BruteForceDetection')
        root.createObject('NewProximityIntersection', name="Proximity", alarmDistance="0.2", contactDistance="0")
        root.createObject('DefaultCollisionGroupManager')
        
        root.createObject('DefaultContactManager', name="Response", response="CompliantContact", responseParams="compliance=0&restitution=0" )
        
        
        root.createObject('CompliantImplicitSolver',stabilization="1")
        root.createObject('SequentialSolver',iterations="10",precision="1e-10")
        root.createObject('LDLTResponse')
                
        
        
        ########################    fixed collider    ########################
        
        fixedNode = root.createChild('fixed')
        fixedNode.createObject('MechanicalObject',template="Rigid",position="0 0 0 0 0 0 1")
        fixedNode.createObject('RigidMass',mass="1",inertia="1 1 1")
        fixedNode.createObject('FixedConstraint',template="Rigid",indices="0")
        fixedNode.createObject('Sphere',contactRestitution=1)

        
        ########################   moving spheres   ########################
        
        
        for c in coefs:
            movingNode = root.createChild('moving'+str(c))
            movingNode.createObject('MechanicalObject',template="Rigid",name="dofs",position="0.5 0 0 0 0 0 1",velocity=str(vel)+" 0 0 0 0 0")
            movingNode.createObject('RigidMass',mass="1",inertia="1 1 1")
            movingNode.createObject('Sphere',contactRestitution=c,group=1)
                

        return 0



class TestController(SofaTest.Controller):

    dofs = []
    
    def initGraph(self,node):
        
        for c in coefs:
            path = '/moving'+str(c)+"/dofs"
            self.dofs.append( node.getObject( path ) )
            
        return 0
                
    def onEndAnimationStep(self, dt):

        for d,c in zip(self.dofs,coefs):
            v = d.velocity[0][0]
            t = -vel*c
            msg = "coef="+str(c)+": simulated="+str(v)+", theoritical="+str(t)
            self.should( abs(v-t) < 1e-10, msg )
            

