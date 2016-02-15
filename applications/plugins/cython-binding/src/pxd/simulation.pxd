# -*- coding: ASCII -*-
from boost cimport intrusive_ptr
from base cimport Base
from node cimport Node, SPtr

cdef extern from "" namespace "sofa::simulation::Simulation": 
    ctypedef intrusive_ptr[Simulation] SimulationSPtr

cdef extern from "../../../../../modules/sofa/simulation/common/Simulation.h" namespace "sofa::simulation": 
    cdef cppclass Simulation(Base):        
        Simulation() except + 
 
        void init(Node* root)
        void initNode(Node* node)
        void animate(Node* root, double dt)
        void reset(Node* root)
        void load(const char* filename)
        void unload(SPtr root)
        
 
cdef extern from "../../../../../modules/sofa/simulation/common/Simulation.h" namespace "sofa::simulation::Simulation":       
        SPtr GetRoot()                
        cdef SimulationSPtr theSimulation  

