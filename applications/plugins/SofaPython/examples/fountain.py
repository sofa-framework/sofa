import Sofa
import sys    
from random import randint, uniform

############################################################################################
# this class sends a script event as soon as the particle has fallen below a certain level
############################################################################################
class Particle(Sofa.PythonScriptController):
    # called once the script is loaded
    def onLoaded(self,node):
        self.myNode = node
        self.particleObject = node.getObject('MecaObject')
    
    # called on each animation step
    def onBeginAnimationStep(self,dt):
        position = self.particleObject.findData('position').value
        if position[0][1] < -5.0:
            self.myNode.getParents()[0].sendScriptEvent('below_floor',self.myNode.name)
        return 0


############################################################################################
# in this sample, a controller spawns particles. It is responsible of deleting them once under the "floor" and re-spawning new ones
# Each particle has a script that checks the particle's altitude and informs the fountain script to delete it through a ScriptEvent
# when it reaches the minimum altitude, so that it can be removed from the scene and respawned.
############################################################################################

class Fountain(Sofa.PythonScriptController):
    
    def createCube(self,parentNode,name,vx,vy,vz,color):
        node = parentNode.createChild(name)

        node.createObject('EulerImplicit')
        node.createObject('CGLinearSolver',iterations=25,tolerance=1.0e-9,threshold=1.0e-9)
        object = node.createObject('MechanicalObject',name='MecaObject',template='Rigid')
        node.createObject('UniformMass',totalMass='1')
        node.createObject('SphereModel',radius='0.5', group='1')

        # VisualNode
        VisuNode = node.createChild('Visu')
        VisuNode.createObject('MeshObjLoader',name='visualMeshLoader',filename='mesh/PokeCube.obj')
        VisuNode.createObject('OglModel',name='Visual',src='@visualMeshLoader',color=color)
        VisuNode.createObject('RigidMapping',input='@..',output='@Visual')

        # apply wanted initial translation
        object.findData('position').value='0 0 0  0 0 0 1'
        object.findData('velocity').value=str(vx)+' '+str(vy)+' '+str(vz)+' 0 0 0'
        
        return node

    
    # called once the script is loaded
    def onLoaded(self,node):
        print 'Fountain.onLoaded called from node '+node.name
        self.rootNode = node
    
    particleCount = 0
    def spawnParticle(self):
        # create the particle, with a random color
        colors=['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        color = colors[randint(0,len(colors) -1)]
        node = self.createCube(self.rootNode,'particle'+str(self.particleCount),uniform(-10,10),uniform(10,30),uniform(-10,10),color)
        self.particleCount+=1
        # add the controller script
        node.createObject('PythonScriptController', filename='fountain.py', classname='Particle')
        return node
     
    # optionnally, script can create a graph...
    def createGraph(self,node):
        print 'Fountain.createGraph called from node '+node.name
        for i in range(1,100):
            node = self.spawnParticle()
        return 0
    
    def onScriptEvent(self,senderNode,eventName,data):
        print 'onScriptEvent eventName=' + eventName + ' sender=' + data
        if eventName=='below_floor':
            self.rootNode.removeChild(self.rootNode.getChild(data))
            node = self.spawnParticle()
            node.init()




