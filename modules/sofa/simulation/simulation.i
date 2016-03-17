%module simulation

%include "std_string.i"

%import "sofa/core/core.i"

%{    
#include <cstdlib>

#include "sofa/simulation/common/Node.h"
#include "sofa/simulation/common/Simulation.h"

#include "sofa/simulation/common/init.h"
#include "sofa/simulation/graph/init.h"
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include "sofa/simulation/graph/DAGSimulation.h"

    void cleanup()
    {
        printf("cleanup!\n");
    }
%}

// TODO put this in a file to be included in sofa .i files
%typemap(out) sofa::simulation::Node::SPtr {
    $result = SWIG_NewPointerObj($1.get(), SWIGTYPE_p_sofa__simulation__Node, 0 );
}

%nodefaultctor;
%nodefaultdtor;

namespace sofa
{

namespace simulation
{

class Node {
public:
    virtual sofa::simulation::Node::SPtr createChild(const std::string& nodeName)=0;
    const sofa::core::objectmodel::BaseContext* getContext() const;
};

class Simulation: public virtual sofa::core::objectmodel::Base {
public:
    static sofa::simulation::Node::SPtr GetRoot();
};

}
}

%init %{
    sofa::simulation::common::init();
    std::atexit(sofa::simulation::common::cleanup);
    sofa::simulation::graph::init();
    std::atexit(sofa::simulation::graph::cleanup);

    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());
    sofa::simulation::getSimulation()->createNewGraph("root");
%}

%pythoncode %{

import sofa.core

# mimic SofaPython Node.createObject method
def Node_createObject(self, type, **kwargs):
    desc = sofa.core.BaseObjectDescription(type, type)
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            desc.setAttribute(key, value)
    return sofa.core.ObjectFactory.CreateObject(self.getContext(), desc)

# turn the function into a method of Node class
Node.createObject=Node_createObject
%}


