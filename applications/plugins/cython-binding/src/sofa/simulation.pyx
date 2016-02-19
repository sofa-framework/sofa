from sofa.basenode cimport _BaseNode 
from sofa.node cimport Node, _Node, _SPtr as _NodeSPtr 
from sofa.simulation cimport _Simulation, GetRoot as _Simulation_GetRoot, theSimulation as _theSimulation

cdef class Simulation:
        """ Main controller of the scene.
            Defines how the scene is inited at the beginning, and updated at each time step.
            Derives from Base in order to use smart pointers and model the parameters as Datas, 
            which makes their edition easy in the GUI.
        """
       
        
        @staticmethod
        def getRoot():
                """Returns the root node of the current simulation."""
                cdef _NodeSPtr tmp = _Simulation_GetRoot()
                n = Node.createNodeFrom(tmp.get())
                return n
                
        @staticmethod
        def init(Node node not None):
                """Initialize the scene graph starting from the provided root node
                        
                   Example of use:
                        Simulation.init( Simulation.getRoot() )  
                """
                _theSimulation.get().init(node.nodeptr) 
                
        @staticmethod
        def initNode(Node node not None):
                """Initialize the node without its context
                        
                   Example of use:
                        Simulation.init( Simulation.getTreeNode("ATreeNode") )  
                """
                _theSimulation.get().initNode(node.nodeptr)
                        
        @staticmethod
        def animate(Node node not None, float dt=0.0):
                """Do one simulation step. The duration of the simulated step 
                   is given in dt. If dt is 0, the dt parameter in the graph 
                   will be used.
                        
                   Example of use:
                        Simulation.animate( Simulation.getRoot(), 0.1 )  
                """
                _theSimulation.get().animate(node.nodeptr, dt)
        
        @staticmethod
        def reset(Node node not None):
                """Reset to initial state. 
                        
                   Example of use:
                        Simulation.reset( Simulation.getRoot() )  
                """
                _theSimulation.get().reset(node.nodeptr)
                
       
