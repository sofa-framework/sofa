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
#include <sofa/simulation/tree/PropagateEventVisitor.h>
#include <sofa/simulation/tree/AnimateVisitor.h>
#include <sofa/simulation/tree/InitVisitor.h>
#include <sofa/simulation/tree/VisualVisitor.h>
#include <sofa/simulation/tree/UpdateMappingVisitor.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{
using core::objectmodel::BaseObject;
using helper::system::thread::CTime;

System::System(const std::string& name)
    : sofa::core::objectmodel::Context()
    , debug_(false), logTime_(false)
{
    totalTime.nVisit = 0;
    totalTime.tNode = 0;
    totalTime.tTree = 0;
    setName(name);
}


System::~System()
{
}

/// Initialize the components
void System::init()
{
    //cerr<<"System::init() begin node "<<getName()<<endl;
    execute<simulation::tree::InitVisitor>();
    //cerr<<"System::init() end node "<<getName()<<endl;
}

/// Do one step forward in time
void System::animate( double dt )
{
    simulation::tree::AnimateVisitor vis(dt);
    //cerr<<"System::animate, start execute"<<endl;
    execute(vis);
    //cerr<<"System::animate, end execute"<<endl;
    execute<simulation::tree::UpdateMappingVisitor>();
}

void System::glDraw()
{
    execute<simulation::tree::VisualUpdateVisitor>();
    execute<simulation::tree::VisualDrawVisitor>();
}





/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
bool System::addObject(BaseObject* obj)
{
    notifyAddObject(obj);
    doAddObject(obj);
    return true;
}

/// Remove an object
bool System::removeObject(BaseObject* obj)
{
    notifyRemoveObject(obj);
    doRemoveObject(obj);
    return true;
}

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
        notifyMoveObject(obj,prev);
        prev->doRemoveObject(obj);
        doAddObject(obj);
    }
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
    notifyAddObject(obj);
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
    actionScheduler.add(dynamic_cast< VisitorScheduler* >(obj));

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

    actionScheduler.remove(dynamic_cast< VisitorScheduler* >(obj));
    // Remove references to this object in time log tables
    if (!objectTime.empty())
    {
        for (std::map<std::string, std::map<core::objectmodel::BaseObject*, ObjectTimer> >::iterator it = objectTime.begin(); it != objectTime.end(); ++it)
        {
            it->second.erase(obj);
        }
    }
}



/// Topology
core::componentmodel::topology::Topology* System::getTopology() const
{
    return this->topology;
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
    return main;
}

/// Mesh Topology (unified interface for both static and dynamic topologies)
core::componentmodel::topology::BaseMeshTopology* System::getMeshTopology() const
{
    return this->meshTopology;
}

/// Shader
core::objectmodel::BaseObject* System::getShader() const
{
    return shader;
}

/// Mechanical Degrees-of-Freedom
core::objectmodel::BaseObject* System::getMechanicalState() const
{
    return this->mechanicalState;
}


void System::setLogTime(bool b)
{
    logTime_=b;
}

System::ctime_t System::getTimeFreq() const
{
    return CTime::getTicksPerSec();
}

void System::resetTime()
{
    totalTime.nVisit = 0;
    totalTime.tNode = 0;
    totalTime.tTree = 0;
    actionTime.clear();
    objectTime.clear();
}

/// Measure start time
System::ctime_t System::startTime() const
{
    if (!getLogTime()) return 0;
    return CTime::getTime();
}

/// Log time spent on an action category and the concerned object
void System::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
}

/// Log time spent given a start time, an action category, and the concerned object
System::ctime_t System::endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj)
{
    if (!getLogTime()) return 0;
    const ctime_t t1 = CTime::getTime();
    const ctime_t t = t1 - t0;
    addTime(t, s, obj);
    return t1;
}

/// Log time spent on an action category, and the concerned object, plus remove the computed time from the parent caller object
void System::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* /*parent*/)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
    //objectTime[s][parent].tObject -= t;
    cerr<<"Warning: System::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent) does not remove the computed time from the parent caller object (parent is ndefined)"<<endl;
}

/// Log time spent given a start time, an action category, and the concerned object
System::ctime_t System::endTime(ctime_t /*t0*/, const std::string& /*s*/, core::objectmodel::BaseObject* /*obj*/, core::objectmodel::BaseObject* /*parent*/)
{
    if (!getLogTime()) return 0;
    const ctime_t t1 = CTime::getTime();
    //const ctime_t t = t1 - t0;
    //addTime(t, s, obj, parent);
    cerr<<"Warning: System::endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent) does not add parent time (parent is ndefined)"<<endl;
    return t1;
}

System* System::setDebug(bool b)
{
    debug_=b;
    return this;
}

bool System::getDebug() const
{
    return debug_;
}




void System::removeControllers()
{
    removeObject(masterSolver);
    typedef Sequence<core::componentmodel::behavior::OdeSolver> Solvers;
    Solvers solverRemove = solver;
    for ( Solvers::iterator i=solverRemove.begin(), iend=solverRemove.end(); i!=iend; i++ )
        removeObject( *i );
}


core::objectmodel::BaseContext* System::getContext()
{
    return this;
}
const core::objectmodel::BaseContext* System::getContext() const
{
    return this;
}


void System::setDefaultVisualContextValue()
{
    if (showVisualModels_.getValue() == -1)            showVisualModels_.setValue(true);
    if (showBehaviorModels_.getValue() == -1)          showBehaviorModels_.setValue(false);
    if (showCollisionModels_.getValue() == -1)         showCollisionModels_.setValue(false);
    if (showBoundingCollisionModels_.getValue() == -1) showBoundingCollisionModels_.setValue(false);
    if (showMappings_.getValue() == -1)                showMappings_.setValue(false);
    if (showMechanicalMappings_.getValue() == -1)      showMechanicalMappings_.setValue(false);
    if (showForceFields_.getValue() == -1)             showForceFields_.setValue(false);
    if (showInteractionForceFields_.getValue() == -1)  showInteractionForceFields_.setValue(false);
    if (showWireFrame_.getValue() == -1)               showWireFrame_.setValue(false);
    if (showNormals_.getValue() == -1)                 showNormals_.setValue(false);
}

void System::initialize()
{
    //cerr<<"System::initialize()"<<endl;

    initVisualContext();
    // Put the OdeSolver, if any, in first position. This makes sure that the OdeSolver component is initialized only when all its sibling and children components are already initialized.
    /// @todo Putting the solver first means that it will be initialized *before* any sibling or childrens. Is that what we want? -- Jeremie A.
    Sequence<BaseObject>::iterator i=object.begin(), iend=object.end();
    for ( ; i!=iend && dynamic_cast<core::componentmodel::behavior::OdeSolver*>(*i)==NULL; i++ ) // find the OdeSolver
    {}
    if ( i!=iend && !object.empty() ) // found
    {
        // put it first
        // BUGFIX 01/12/06 (Jeremie A.): do not modify the order of the other objects
        // object.swap( i, object.begin() );
        while (i!=object.begin())
        {
            Sequence<BaseObject>::iterator i2 = i;
            --i;
            object.swap(i, i2);
        }
    }

    //
    updateSimulationContext();

    // this is now done by the InitVisitor
    //for (Sequence<System>::iterator it = child.begin(); it != child.end(); it++) {
    //    (*it)->init();
    //}
}

void System::updateContext()
{
    //if( debug_ ) cerr<<"System::updateContext, node = "<<getName()<<", incoming context = "<< *this->getContext() << endl;

    // Apply local modifications to the context
    if (getLogTime())
    {
        for ( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
            //cerr<<"System::updateContext, modified by node = "<<contextObject[i]->getName()<< endl;
        }
    }
    else
    {
        for ( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
            //cerr<<"System::updateContext, modified by node = "<<contextObject[i]->getName()<<endl;
        }
    }
//	if( !mechanicalModel.empty() ) {
//		mechanicalModel->updateContext(&context_);
//	}


    // project the gravity to the local coordinate system
    /*        getContext()->setGravity( getContext()->getLocalFrame().backProjectVector(getContext()->getWorldGravity()) );*/

    if ( debug_ ) std::cerr<<"System::updateContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}

void System::updateSimulationContext()
{
    // Apply local modifications to the context
    if (getLogTime())
    {
        for ( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    else
    {
        for ( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    if ( debug_ ) std::cerr<<"System::updateSimulationContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}

void System::updateVisualContext(int/* FILTER*/)
{
    // Apply local modifications to the context
    if (getLogTime())
    {
        for ( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    else
    {
        for ( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    if ( debug_ ) std::cerr<<"System::updateVisualContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}

/// Execute a recursive action starting from this node
void System::executeVisitor(Visitor* action)
{
    if (!this->is_activated.getValue()) return;
    if (actionScheduler)
        actionScheduler->executeVisitor(this,action);
    else
        doExecuteVisitor(action);
}

/// Propagate an event
void System::propagateEvent( core::objectmodel::Event* event )
{
    simulation::tree::PropagateEventVisitor act(event);
    this->executeVisitor(&act);
}








}

}


