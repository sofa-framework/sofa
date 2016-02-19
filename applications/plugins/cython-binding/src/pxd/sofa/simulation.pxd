# -*- coding: ASCII -*-
from boost cimport intrusive_ptr
from base cimport _Base
from node cimport _Node, _SPtr as _NodeSPtr

cdef extern from "" namespace "sofa::simulation::Simulation": 
    ctypedef intrusive_ptr[_Simulation] _SPtr "sofa::simulation::Simulation::SPtr"

cdef extern from "../../../../../modules/sofa/simulation/common/Simulation.h" namespace "sofa::simulation": 
    cdef cppclass _Simulation "sofa::simulation::Simulation" (_Base):        
        _Simulation() except + 
 
        void init(_Node* root)
        void initNode(_Node* node)
        void animate(_Node* root, double dt)
        void reset(_Node* root)
        void load(const char* filename)
        void unload(_SPtr root)
        
 
cdef extern from "../../../../../modules/sofa/simulation/common/Simulation.h" namespace "sofa::simulation::Simulation":       
        _SPtr GetRoot()                
        cdef _SPtr theSimulation  

#cdef class Simulation:
#        """ Main controller of the scene.
#            Defines how the scene is inited at the beginning, and updated at each time step.
#            Derives from Base in order to use smart pointers and model the parameters as Datas, 
#            which makes their edition easy in the GUI.
#        """

