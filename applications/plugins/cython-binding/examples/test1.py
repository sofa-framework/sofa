# -*- coding: UTF8 -*-
import Sofa
from sofa import *
#from sofacython import *
import ipshell
import timeit

def testBasicAPI():
        r = Simulation.getRoot()   # The current root of the simulation
       
        # Accessing data fields
        print("Data fields are: "+ str(r.getDataNames()))
        
        print("Gravity data field is "+str(r.gravity))
        r.gravity[0] = [2,3,4]
        print("New value is "+str(r.gravity))
        
        
        # Manipulating the scene graph
        print("================= Scenegraph manipulation ========================")
        n = r.createChild("MyChildNode") 
        print("created a new node with name: "+n.getName()) 
       
        o = n.createObject("MechanicalObject", name="myMechanical")
        o.init()
        print("created a new object of class: "+o.getTypeName()) 

        try:
                n.createObject("MechanicalObjectDDD", name="myMechanical")
        except Exception:
                print("Unable to create the requested mechanical object you are a silly person")                 

        # Manipulating concrete objects through data fields
        print("================= Visual Snake manipulation ========================")
        snake = r.getObject("VisualSnake")
        if snake != None:
                print("Data fields are: "+ str(snake.getDataNames()))        
                print("Number of vertices:"+str(len( snake.vertices )))
                for i in range(0, len(snake.vertices)):
                        snake.vertices[i] = [1,2,3]
                        
                sp = snake.position
                for j in range(0, len(snake.position)):
                        sp[j] = [1,2,3]
                snake.reinit()
                
        # Manipulating concrete objects through dedicated binding 
        print("================= BaseMechanicalState manipulation ========================")
        b = BaseMechanicalState(o)  # Downcast from BaseObject to BaseMechanicalState to have access to the 
                                   # dedicated part of the API
        print("Data fields are: "+ str(r.getDataNames()))        
        print("X: ["),                          
        b.resize(10)
        for i in range(0, len(b)):
                print(str(b.getPX(i))+", "),
        
        print("]") 
        
r = None
dp = []
def doSofaBench():
        s = r.getObject("VisualSnake")
        a = None
        #print("Size: "+str(len(s.position)))
        
        #for i in range(0, len(s.position)):
        #        #a = s.position[i]
        #        s.position[i] = [i, i, i] 
        
        #p = s.position
        #for i in range(0,len(p)):
        #        p[i] = [i, i, i]
        #s.position = p
        s.position = dp 
        
        return a 
        
def doCythonBench():
        global dp 
        
        s = r.getObject("VisualSnake")
        a = None
        #print("Size: "+str(len(s.position)))
        for i in range(0, len(dp)):
                s.position[i] = dp[i] 
        return a 

def doCythonBench2():
        global dp 
        
        s = r.getObject("VisualSnake")
        a = None
        #print("Size: "+str(len(s.position)))
        p = s.position
        for i in range(0, len(dp)):
                p[i] = dp[i]
        return a 
        
def doCythonBench3():
        global dp 
        s = r.getObject("VisualSnake")
        a = None
        s.position = dp  
        return a
                
class Test1(Sofa.PythonScriptController):
        def initGraph(self,node):
		self.node = node

        def doBench(self):
                global r, dp
                dp = []
                r = self.node.getRoot()
                s = r.getObject("VisualSnake")
                
                for i in range(0,len(s.position)):
                        dp.append([i, i, i])
                 
                r = self.node.getRoot()
                print("SOFA TIME: "+str(timeit.timeit(doSofaBench, number=50)))                
                
                r = Simulation.getRoot() 
                print("Cython TIME: "+str(timeit.timeit(doCythonBench, number=50)))                
                
                r = Simulation.getRoot() 
                print("Cython2 TIME: "+str(timeit.timeit(doCythonBench2, number=50)))      
                
                r = Simulation.getRoot() 
                print("Cython3 TIME: "+str(timeit.timeit(doCythonBench3, number=50)))           

        def onKeyPressed(self, k):
                if k == 'C':
                        ipshell.start()
                else:
                        self.doBench()
                        #testBasicAPI() 
