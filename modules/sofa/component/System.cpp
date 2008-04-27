//
// C++ Implementation: System
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "System.h"

namespace sofa
{

namespace component
{

System::System()
    : sofa::core::objectmodel::Context()
{
}


System::~System()
{
}

using core::objectmodel::BaseObject;


/// Move an object from another node
void System::moveObject(BaseObject* obj)
{
    System* prev = dynamic_cast<System*>(obj->getContext());
    if (prev==NULL)
    {
        obj->getContext()->removeObject(obj);
        addObject(obj);
    }
    else
    {
        prev->doRemoveObject(obj);
        doAddObject(obj);
    }
}

/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
bool System::addObject(BaseObject* obj)
{
    doAddObject(obj);
    return true;
}

/// Remove an object
bool System::removeObject(BaseObject* obj)
{
    doRemoveObject(obj);
    return true;
}

/// Find an object given its name
core::objectmodel::BaseObject* System::getObject(const std::string& name) const
{
    for (ObjectIterator it = object.begin(), itend = object.end(); it != itend; ++it)
        if ((*it)->getName() == name)
            return *it;
    return NULL;
}



/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
void System::doAddObject(BaseObject* obj)
{
    obj->setContext(this);
    object.add(obj);
    masterSolver.add(dynamic_cast< core::componentmodel::behavior::MasterSolver* >(obj));
    solver.add(dynamic_cast< core::componentmodel::behavior::OdeSolver* >(obj));
    mechanicalState.add(dynamic_cast< core::componentmodel::behavior::BaseMechanicalState* >(obj));
    if (!mechanicalMapping.add(dynamic_cast< core::componentmodel::behavior::BaseMechanicalMapping* >(obj)))
        mapping.add(dynamic_cast< core::BaseMapping* >(obj));
    mass.add(dynamic_cast< core::componentmodel::behavior::BaseMass* >(obj));
    topology.add(dynamic_cast< core::componentmodel::topology::Topology* >(obj));
    basicTopology.add(dynamic_cast< core::componentmodel::topology::BaseTopology* >(obj));
    meshTopology.add(dynamic_cast< core::componentmodel::topology::BaseMeshTopology* >(obj));
    shader.add(dynamic_cast< sofa::core::Shader* >(obj));

    if (!interactionForceField.add(dynamic_cast< core::componentmodel::behavior::InteractionForceField* >(obj)))
        forceField.add(dynamic_cast< core::componentmodel::behavior::BaseForceField* >(obj));
    constraint.add(dynamic_cast< core::componentmodel::behavior::BaseConstraint* >(obj));
    behaviorModel.add(dynamic_cast< core::BehaviorModel* >(obj));
    visualModel.add(dynamic_cast< core::VisualModel* >(obj));
    collisionModel.add(dynamic_cast< core::CollisionModel* >(obj));
    contextObject.add(dynamic_cast< core::objectmodel::ContextObject* >(obj));
    collisionPipeline.add(dynamic_cast< core::componentmodel::collision::Pipeline* >(obj));
}

/// Remove an object
void System::doRemoveObject(BaseObject* obj)
{
    if (obj->getContext()==this)
    {
        obj->setContext(NULL);
    }
    object.remove(obj);
    masterSolver.remove(dynamic_cast< core::componentmodel::behavior::MasterSolver* >(obj));
    solver.remove(dynamic_cast< core::componentmodel::behavior::OdeSolver* >(obj));
    mechanicalState.remove(dynamic_cast< core::componentmodel::behavior::BaseMechanicalState* >(obj));
    mechanicalMapping.remove(dynamic_cast< core::componentmodel::behavior::BaseMechanicalMapping* >(obj));
    mass.remove(dynamic_cast< core::componentmodel::behavior::BaseMass* >(obj));
    topology.remove(dynamic_cast< core::componentmodel::topology::Topology* >(obj));
    basicTopology.remove(dynamic_cast< core::componentmodel::topology::BaseTopology* >(obj));
    meshTopology.remove(dynamic_cast< core::componentmodel::topology::BaseMeshTopology* >(obj));
    shader.remove(dynamic_cast<sofa::core::Shader* >(obj));

    forceField.remove(dynamic_cast< core::componentmodel::behavior::BaseForceField* >(obj));
    interactionForceField.remove(dynamic_cast< core::componentmodel::behavior::InteractionForceField* >(obj));
    constraint.remove(dynamic_cast< core::componentmodel::behavior::BaseConstraint* >(obj));
    mapping.remove(dynamic_cast< core::BaseMapping* >(obj));
    behaviorModel.remove(dynamic_cast< core::BehaviorModel* >(obj));
    visualModel.remove(dynamic_cast< core::VisualModel* >(obj));
    collisionModel.remove(dynamic_cast< core::CollisionModel* >(obj));
    contextObject.remove(dynamic_cast<core::objectmodel::ContextObject* >(obj));
    collisionPipeline.remove(dynamic_cast< core::componentmodel::collision::Pipeline* >(obj));
}



/// Topology
core::componentmodel::topology::Topology* System::getTopology() const
{
    // return this->topology;
    // CHANGE 12/01/06 (Jeremie A.): Inherit parent topology if no local topology is defined
    if (this->topology)
        return this->topology;
    else
        return NULL;
}

/// Dynamic Topology
core::componentmodel::topology::BaseTopology* System::getMainTopology() const
{
    core::componentmodel::topology::BaseTopology *main=0;
    unsigned int i;
    for (i=0; i<basicTopology.size(); ++i)
    {
        if (basicTopology[i]->isMainTopology()==true)
            main=basicTopology[i];
    }
    // return main;
    // CHANGE 12/01/06 (Jeremie A.): Inherit parent topology if no local topology is defined
    if (main)
        return main;
    else
        return NULL;
}

/// Mesh Topology (unified interface for both static and dynamic topologies)
core::componentmodel::topology::BaseMeshTopology* System::getMeshTopology() const
{
    if (this->meshTopology)
        return this->meshTopology;
    else
        return NULL;
}

/// Shader
core::objectmodel::BaseObject* System::getShader() const
{
    if (shader)
        return shader;
    else
        return NULL;
}

/// Mechanical Degrees-of-Freedom
core::objectmodel::BaseObject* System::getMechanicalState() const
{
    // return this->mechanicalModel;
    // CHANGE 12/01/06 (Jeremie A.): Inherit parent mechanical model if no local model is defined
    if (this->mechanicalState)
        return this->mechanicalState;
    else
        return NULL;
}





}

}
