# -*- coding: UTF8 -*-
import Sofa
import sofacython
from sofacython import *
import ipshell

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
        
       

class Test1(Sofa.PythonScriptController):
        def onKeyPressed(self, k):
                if k == 'c':
                        ipshell.start()
                else:
                        testBasicAPI() 
