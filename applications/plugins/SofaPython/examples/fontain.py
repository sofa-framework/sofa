import Sofa

import sys
for p in sys.path:
    print p
#for m in sys.modules:
#    print m
    
print __name__    

import random

############################################################################################
# in this sample, a controller spawns particles and is responsible to delete them when necessary, then re-spawn others
# Each particle has a script itself that check the particle's altitude an sends a message to the fontain script
# when it reachs the minimum altitude, so that it can be removed from the scene.
############################################################################################

class Fontain(Sofa.PythonScriptController):


    def createCube(self,parentNode,name,x,y,z,vx,vy,vz,color):
        node = parentNode.createChild(name)

        node.createObject('EulerImplicit')
        node.createObject('CGLinearSolver',iterations=25,tolerance=1.0e-9,threshold=1.0e-9)
        object = node.createObject('MechanicalObject',name='MecaObject',template='Rigid')
        node.createObject('UniformMass',totalmass=100)

        # VisualNode
        VisuNode = node.createChild('Visu')
        VisuNode.createObject('OglModel',name='Visual',fileMesh='mesh/PokeCube.obj',color=color)
        VisuNode.createObject('RigidMapping',input='@..',output='@Visual')

        # apply wanted initial translation
        #object.applyTranslation(x,y,z)
        object.findData('position').value=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1'
        object.findData('velocity').value=str(vx)+' '+str(vy)+' '+str(vz)+' 0 0 0'
        
        return node

    
    # called once the script is loaded
    def onLoaded(self,node):
        print 'Fontain.onLoaded called from node '+node.name
        self.rootNode = node
    
    particleCount = 0
    def spawnParticle(self):
        # create the particle, with a random color
        color='red'
        colorRandom = random.randint(1,6)
        if colorRandom==1:
            color = 'red'
        if colorRandom==2:
            color = 'green'
        if colorRandom==3:
            color = 'blue'
        if colorRandom==4:
            color = 'yellow'
        if colorRandom==5:
            color = 'cyan'
        if colorRandom==6:
            color = 'magenta'
        node = self.createCube(self.rootNode,'particle'+str(self.particleCount),0,0,0,random.uniform(-10,10),random.uniform(10,30),random.uniform(-10,10),color)
        self.particleCount+=1
        # add the controller script
        node.createObject('PythonScriptController', filename='fontain.py', classname='Particle')
        return node
     
    # optionnally, script can create a graph...
    def createGraph(self,node):
        print 'Fontain.createGraph called from node '+node.name    
        for i in range(1,100):
            node = self.spawnParticle()
            node.init()
        return 0
    
    def onScriptEvent(self,senderNode,eventName,data):
        print 'onScriptEvent eventName='+eventName+' data='+str(data)+' sender='+senderNode.name
        if eventName=='below_floor':
            self.rootNode.removeChild(senderNode)
            self.spawnParticle()





############################################################################################
# this class sends a script event as soon as the particle has fallen below a certain level
############################################################################################
class Particle(Sofa.PythonScriptController):
    # called once the script is loaded
    def onLoaded(self,node):
        self.myNode = node
        self.particleObject=node.getObject('MecaObject')
    
    # called on each animation step
    def onBeginAnimationStep(self,dt):
        position = self.particleObject.findData('position').value
        if position[0][1]<-5.0:
            self.myNode.sendScriptEvent('below_floor',0)
        return 0
