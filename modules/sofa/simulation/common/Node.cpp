/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Implementation: Node
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/xml/NodeElement.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/DeactivatedNodeVisitor.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>

#include <sofa/helper/Factory.inl>
#include <sofa/simulation/common/xml/Element.inl>
#include <iostream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>

//#define DEBUG_VISITOR

namespace sofa
{

namespace simulation
{
using std::cerr;
using std::endl;
using core::objectmodel::BaseNode;
using core::objectmodel::BaseObject;
using helper::system::thread::CTime;

Node::Node(const std::string& name)
    : core::objectmodel::BaseNode()
    , sofa::core::objectmodel::Context()
    , debug_(false), logTime_(false)
    , depend(initData(&depend,"depend","Dependencies between the nodes.\nname 1 name 2 name3 name4 means that name1 must be initialized before name2 and name3 before name4"))
{
    _context = this;
    totalTime.nVisit = 0;
    totalTime.tNode = 0;
    totalTime.tTree = 0;
    setName(name);
}


Node::~Node()
{
}

/// Initialize the components of this node and all the nodes which depend on it.
void Node::init(const core::ExecParams* params)
{
//     cerr<<"Node::init() begin node "<<getName()<<endl;
    execute<simulation::InitVisitor>(params);
//     cerr<<"Node::init() end node "<<getName()<<endl;
}

/// ReInitialize the components of this node and all the nodes which depend on it.
void Node::reinit(const core::ExecParams* params)
{
    sofa::simulation::DeactivationVisitor deactivate(params, isActive());
    deactivate.execute( this );
}

/// Do one step forward in time
void Node::animate(const core::ExecParams* params /* PARAMS FIRST */, double dt)
{
    simulation::AnimateVisitor vis(params /* PARAMS FIRST */, dt);
    //cerr<<"Node::animate, start execute"<<endl;
    execute(vis);
    //cerr<<"Node::animate, end execute"<<endl;
}

void Node::glDraw(core::visual::VisualParams* vparams)
{
    execute<simulation::VisualUpdateVisitor>(vparams);
    execute<simulation::VisualDrawVisitor>(vparams);
}


/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
bool Node::addObject(BaseObject::SPtr obj)
{
    notifyAddObject(obj);
    doAddObject(obj);
    return true;
}

/// Remove an object
bool Node::removeObject(BaseObject::SPtr obj)
{
    notifyRemoveObject(obj);
    doRemoveObject(obj);
    return true;
}

/// Move an object from another node
void Node::moveObject(BaseObject::SPtr obj)
{
    Node* prev = dynamic_cast<Node*>(obj->getContext());
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



void Node::notifyAddChild(Node::SPtr node)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->addChild(this, node.get());
}


void Node::notifyRemoveChild(Node::SPtr node)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->removeChild(this, node.get());
}


void Node::notifyMoveChild(Node::SPtr node, Node* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveChild(prev, this, node.get());
}


void Node::notifyAddObject(core::objectmodel::BaseObject::SPtr obj)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->addObject(this, obj.get());
}

void Node::notifyRemoveObject(core::objectmodel::BaseObject::SPtr obj)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->removeObject(this, obj.get());
}

void Node::notifyMoveObject(core::objectmodel::BaseObject::SPtr obj, Node* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveObject(prev, this, obj.get());
}


void Node::addListener(MutationListener* obj)
{
    // make sure we don't add the same listener twice
    Sequence< MutationListener >::iterator it = listener.begin();
    while (it != listener.end() && (*it)!=obj)
        ++it;
    if (it == listener.end())
        listener.add(obj);
}

void Node::removeListener(MutationListener* obj)
{
    listener.remove(obj);
}


/// Find an object given its name
core::objectmodel::BaseObject* Node::getObject(const std::string& name) const
{
    for (ObjectIterator it = object.begin(), itend = object.end(); it != itend; ++it)
        if ((*it)->getName() == name)
            return it->get();
    return NULL;
}



/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
void Node::doAddObject(BaseObject::SPtr sobj)
{
    notifyAddObject(sobj);
    sobj->setContext(this);
    object.add(sobj);
    BaseObject* obj = sobj.get();
    int inserted=0;
    inserted+= animationManager.add(dynamic_cast< core::behavior::BaseAnimationLoop* >(obj));
    inserted+= solver.add(dynamic_cast< core::behavior::OdeSolver* >(obj));
    inserted+= linearSolver.add(dynamic_cast< core::behavior::LinearSolver* >(obj));
    inserted+= constraintSolver.add(dynamic_cast< core::behavior::ConstraintSolver* >(obj));
    inserted+= state.add(dynamic_cast< core::BaseState* >(obj));
    inserted+= mechanicalState.add(dynamic_cast< core::behavior::BaseMechanicalState* >(obj));
    core::BaseMapping* bmap = dynamic_cast< core::BaseMapping* >(obj);
    bool isMechanicalMapping = false;
    if(bmap)
    {
        isMechanicalMapping = bmap->isMechanical();
        if(isMechanicalMapping)
            inserted += mechanicalMapping.add(bmap);
        else
            inserted += mapping.add(bmap);
    }

    inserted+= mass.add(dynamic_cast< core::behavior::BaseMass* >(obj));
    inserted+= topology.add(dynamic_cast< core::topology::Topology* >(obj));
    inserted+= meshTopology.add(dynamic_cast< core::topology::BaseMeshTopology* >(obj));
    inserted+= shader.add(dynamic_cast< sofa::core::visual::Shader* >(obj));

    bool isInteractionForceField = interactionForceField.add(dynamic_cast< core::behavior::BaseInteractionForceField* >(obj));
    inserted+= isInteractionForceField;
    if (!isInteractionForceField)
        forceField.add(dynamic_cast< core::behavior::BaseForceField* >(obj));
    inserted+= projectiveConstraintSet.add(dynamic_cast< core::behavior::BaseProjectiveConstraintSet* >(obj));
    inserted+= constraintSet.add(dynamic_cast< core::behavior::BaseConstraintSet* >(obj));
    inserted+= behaviorModel.add(dynamic_cast< core::BehaviorModel* >(obj));
    inserted+= visualModel.add(dynamic_cast< core::visual::VisualModel* >(obj));
    inserted+= visualManager.add(dynamic_cast< core::visual::VisualManager* >(obj));
    inserted+= collisionModel.add(dynamic_cast< core::CollisionModel* >(obj));
    inserted+= contextObject.add(dynamic_cast< core::objectmodel::ContextObject* >(obj));
    inserted+= configurationSetting.add(dynamic_cast< core::objectmodel::ConfigurationSetting* >(obj));
    inserted+= collisionPipeline.add(dynamic_cast< core::collision::Pipeline* >(obj));
    inserted+= actionScheduler.add(dynamic_cast< VisitorScheduler* >(obj));

    if ( inserted==0 )
    {
        //cerr<<"Node::doAddObject, object "<<obj->getName()<<" is unsorted"<<endl;
        unsorted.add(obj);
    }

}

/// Remove an object
void Node::doRemoveObject(BaseObject::SPtr sobj)
{
    if (sobj->getContext()==this)
    {
        sobj->setContext(NULL);
    }
    object.remove(sobj);
    BaseObject* obj = sobj.get();
    animationManager.remove(dynamic_cast< core::behavior::BaseAnimationLoop* >(obj));
    solver.remove(dynamic_cast< core::behavior::OdeSolver* >(obj));
    linearSolver.remove(dynamic_cast< core::behavior::LinearSolver* >(obj));
    constraintSolver.remove(dynamic_cast< core::behavior::ConstraintSolver* >(obj));
    state.remove(dynamic_cast< core::BaseState* >(obj));
    mechanicalState.remove(dynamic_cast< core::behavior::BaseMechanicalState* >(obj));
    mechanicalMapping.remove(dynamic_cast< core::BaseMapping* >(obj));
    mass.remove(dynamic_cast< core::behavior::BaseMass* >(obj));
    topology.remove(dynamic_cast< core::topology::Topology* >(obj));
    meshTopology.remove(dynamic_cast< core::topology::BaseMeshTopology* >(obj));
    shader.remove(dynamic_cast<sofa::core::visual::Shader* >(obj));

    forceField.remove(dynamic_cast< core::behavior::BaseForceField* >(obj));
    interactionForceField.remove(dynamic_cast< core::behavior::BaseInteractionForceField* >(obj));
    projectiveConstraintSet.remove(dynamic_cast< core::behavior::BaseProjectiveConstraintSet* >(obj));
    constraintSet.remove(dynamic_cast< core::behavior::BaseConstraintSet* >(obj));
    mapping.remove(dynamic_cast< core::BaseMapping* >(obj));
    behaviorModel.remove(dynamic_cast< core::BehaviorModel* >(obj));
    visualModel.remove(dynamic_cast< core::visual::VisualModel* >(obj));
    visualManager.remove(dynamic_cast< core::visual::VisualManager* >(obj));
    collisionModel.remove(dynamic_cast< core::CollisionModel* >(obj));
    contextObject.remove(dynamic_cast<core::objectmodel::ContextObject* >(obj));
    configurationSetting.remove(dynamic_cast<core::objectmodel::ConfigurationSetting* >(obj));
    collisionPipeline.remove(dynamic_cast< core::collision::Pipeline* >(obj));

    actionScheduler.remove(dynamic_cast< VisitorScheduler* >(obj));

    unsorted.remove(obj);
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
core::topology::Topology* Node::getTopology() const
{
    // return this->topology;
    if (this->topology)
        return this->topology;
    else
        return get<core::topology::Topology>();
}

/// Mesh Topology (unified interface for both static and dynamic topologies)
core::topology::BaseMeshTopology* Node::getMeshTopology() const
{
    if (this->meshTopology)
        return this->meshTopology;
    else
        return get<core::topology::BaseMeshTopology>();
}

/// Shader
core::objectmodel::BaseObject* Node::getShader() const
{
    if (shader)
        return shader;
    else
        return get<core::visual::Shader>();
}

/// Degrees-of-Freedom
core::objectmodel::BaseObject* Node::getState() const
{
    // return this->state;
    if (this->state)
        return this->state;
    else
        return get<core::BaseState>();
}

/// Mechanical Degrees-of-Freedom
core::objectmodel::BaseObject* Node::getMechanicalState() const
{
    // return this->mechanicalModel;
    if (this->mechanicalState)
        return this->mechanicalState;
    else
        return get<core::behavior::BaseMechanicalState>();
}



/// Find a child node given its name
Node* Node::getChild(const std::string& name) const
{
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
        if ((*it)->getName() == name)
            return it->get();
    return NULL;
}

/// Get a descendant node given its name
Node* Node::getTreeNode(const std::string& name) const
{
    Node* result = NULL;
    result = getChild(name);
    for (ChildIterator it = child.begin(), itend = child.end(); result == NULL && it != itend; ++it)
        result = (*it)->getTreeNode(name);
    return result;
}

/// Get parent node (or NULL if no hierarchy or for root node)
const sofa::core::objectmodel::BaseNode::Children Node::getChildren() const
{
    Children list_children;
    list_children.reserve(child.size());
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
        list_children.push_back(it->get());
    return list_children;
}



void Node::setLogTime(bool b)
{
    logTime_=b;
}

Node::ctime_t Node::getTimeFreq() const
{
    return CTime::getTicksPerSec();
}

void Node::resetTime()
{
    totalTime.nVisit = 0;
    totalTime.tNode = 0;
    totalTime.tTree = 0;
    actionTime.clear();
    objectTime.clear();
}

/// Measure start time
Node::ctime_t Node::startTime() const
{
    if (!getLogTime()) return 0;
    return CTime::getTime();
}

/// Log time spent on an action category and the concerned object
void Node::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
}

/// Log time spent given a start time, an action category, and the concerned object
Node::ctime_t Node::endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj)
{
    if (!getLogTime()) return 0;
    const ctime_t t1 = CTime::getTime();
    const ctime_t t = t1 - t0;
    addTime(t, s, obj);
    return t1;
}

/// Log time spent on an action category, and the concerned object, plus remove the computed time from the parent caller object
void Node::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* /*parent*/)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
    //objectTime[s][parent].tObject -= t;
    cerr<<"Warning: Node::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent) does not remove the computed time from the parent caller object (parent is ndefined)"<<endl;
}

/// Log time spent given a start time, an action category, and the concerned object
Node::ctime_t Node::endTime(ctime_t /*t0*/, const std::string& /*s*/, core::objectmodel::BaseObject* /*obj*/, core::objectmodel::BaseObject* /*parent*/)
{
    if (!getLogTime()) return 0;
    const ctime_t t1 = CTime::getTime();
    //const ctime_t t = t1 - t0;
    //addTime(t, s, obj, parent);
    cerr<<"Warning: Node::endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent) does not add parent time (parent is ndefined)"<<endl;
    return t1;
}

Node* Node::setDebug(bool b)
{
    debug_=b;
    return this;
}

bool Node::getDebug() const
{
    return debug_;
}




void Node::removeControllers()
{
    removeObject(*animationManager.begin());
    typedef Sequence<core::behavior::OdeSolver> Solvers;
    Solvers solverRemove = solver;
    for ( Solvers::iterator i=solverRemove.begin(), iend=solverRemove.end(); i!=iend; i++ )
        removeObject( *i );
}


core::objectmodel::BaseContext* Node::getContext()
{
    return _context;
}
const core::objectmodel::BaseContext* Node::getContext() const
{
    return _context;
}

// void Node::setContext( core::objectmodel::BaseContext* c )
// {
//     _context=c;
// 	for( ObjectIterator i=object.begin(), iend=object.end(); i!=iend; i++ )
// 		(*i)->setContext(c);
// }


void Node::setDefaultVisualContextValue()
{
    /// @TODO: This method is now broken because getShow*() methods never return -1
    /*
        if (getShowVisualModels() == -1)            setShowVisualModels(true);
        if (getShowBehaviorModels() == -1)          setShowBehaviorModels(false);
        if (getShowCollisionModels() == -1)         setShowCollisionModels(false);
        if (getShowBoundingCollisionModels() == -1) setShowBoundingCollisionModels(false);
        if (getShowMappings() == -1)                setShowMappings(false);
        if (getShowMechanicalMappings() == -1)      setShowMechanicalMappings(false);
        if (getShowForceFields() == -1)             setShowForceFields(false);
        if (getShowInteractionForceFields() == -1)  setShowInteractionForceFields(false);
        if (getShowWireFrame() == -1)               setShowWireFrame(false);
        if (getShowNormals() == -1)                 setShowNormals(false);
    #ifdef SOFA_SMP
        if (showProcessorColor_.getValue() == -1)                 showProcessorColor_.setValue(false);
    #endif
    */
}

void Node::initialize()
{
    //cerr<<"Node::initialize()"<<endl;

    initVisualContext();
    sortComponents();
//     // Put the OdeSolver, if any, in first position. This makes sure that the OdeSolver component is initialized only when all its sibling and children components are already initialized.
//     /// @todo Putting the solver first means that it will be initialized *before* any sibling or childrens. Is that what we want? -- Jeremie A.
//     Sequence<BaseObject>::iterator i=object.begin(), iend=object.end();
//     for ( ; i!=iend && dynamic_cast<core::behavior::OdeSolver*>(*i)==NULL; i++ ) // find the OdeSolver
//         {}
//     if ( i!=iend && !object.empty() ) // found
//     {
//         // put it first
//         // BUGFIX 01/12/06 (Jeremie A.): do not modify the order of the other objects
//         // object.swap( i, object.begin() );
//         while (i!=object.begin())
//         {
//             Sequence<BaseObject>::iterator i2 = i;
//             --i;
//             object.swap(i, i2);
//         }
//     }

    //
    updateSimulationContext();

    // this is now done by the InitVisitor
    //for (Sequence<Node>::iterator it = child.begin(); it != child.end(); it++) {
    //    (*it)->init();
    //}
}

void Node::updateContext()
{
    updateSimulationContext();
    updateVisualContext();
    if ( debug_ ) std::cerr<<"Node::updateContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}

void Node::updateSimulationContext()
{
    for ( unsigned i=0; i<contextObject.size(); ++i )
    {
        contextObject[i]->init();
        contextObject[i]->apply();
//       cerr<<"Node::updateContext, modified by node = "<<contextObject[i]->getName()<<endl;
    }
}

void Node::updateVisualContext()
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
    if ( debug_ ) std::cerr<<"Node::updateVisualContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}

/// Execute a recursive action starting from this node
void Node::executeVisitor(Visitor* action)
{
    if (!this->isActive()) return;

#ifdef DEBUG_VISITOR
    static int level = 0;
    for (int i=0; i<level; ++i) std::cerr << ' ';
    std::cerr << ">" << decodeClassName(typeid(*action)) << " on " << this->getPathName();
//     if (MechanicalVisitor* v = dynamic_cast<MechanicalVisitor*>(action))
    if (!action->getInfos().empty())
        std::cerr << "  : " << action->getInfos();
    std::cerr << std::endl;
    ++level;
#endif

    if (actionScheduler)
        actionScheduler->executeVisitor(this,action);
    else
        doExecuteVisitor(action);

#ifdef DEBUG_VISITOR
    --level;
    for (int i=0; i<level; ++i) std::cerr << ' ';
    std::cerr << "<" << decodeClassName(typeid(*action)) << " on " << this->getPathName();
    std::cerr << std::endl;
#endif

}

/// Propagate an event
void Node::propagateEvent(const core::ExecParams* params /* PARAMS FIRST */, core::objectmodel::Event* event)
{
    simulation::PropagateEventVisitor act(params /* PARAMS FIRST */, event);
    this->executeVisitor(&act);
}





void Node::printComponents()
{
    using namespace sofa::core::behavior;
    using core::BaseMapping;
    using core::topology::Topology;
    using core::topology::BaseTopology;
    using core::topology::BaseMeshTopology;
    using core::visual::Shader;
    using core::BehaviorModel;
    using core::visual::VisualModel;
    using core::CollisionModel;
    using core::objectmodel::ContextObject;
    using core::collision::Pipeline;

    cerr<<"BaseAnimationLoop: ";
    for ( Single<BaseAnimationLoop>::iterator i=animationManager.begin(), iend=animationManager.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"OdeSolver: ";
    for ( Sequence<OdeSolver>::iterator i=solver.begin(), iend=solver.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"LinearSolver: ";
    for ( Sequence<LinearSolver>::iterator i=linearSolver.begin(), iend=linearSolver.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"ConstraintSolver: ";
    for ( Sequence<ConstraintSolver>::iterator i=constraintSolver.begin(), iend=constraintSolver.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"InteractionForceField: ";
    for ( Sequence<BaseInteractionForceField>::iterator i=interactionForceField.begin(), iend=interactionForceField.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"ForceField: ";
    for ( Sequence<BaseForceField>::iterator i=forceField.begin(), iend=forceField.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"State: ";
    for ( Single<BaseState>::iterator i=state.begin(), iend=state.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"MechanicalState: ";
    for ( Single<BaseMechanicalState>::iterator i=mechanicalState.begin(), iend=mechanicalState.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"Mechanical Mapping: ";
    for ( Single<BaseMapping>::iterator i=mechanicalMapping.begin(), iend=mechanicalMapping.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"Mapping: ";
    for ( Sequence<BaseMapping>::iterator i=mapping.begin(), iend=mapping.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"Topology: ";
    for ( Single<Topology>::iterator i=topology.begin(), iend=topology.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"MeshTopology: ";
    for ( Single<BaseMeshTopology>::iterator i=meshTopology.begin(), iend=meshTopology.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"Shader: ";
    for ( Single<Shader>::iterator i=shader.begin(), iend=shader.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"ProjectiveConstraintSet: ";
    for ( Sequence<BaseProjectiveConstraintSet>::iterator i=projectiveConstraintSet.begin(), iend=projectiveConstraintSet.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"ConstraintSet: ";
    for ( Sequence<BaseConstraintSet>::iterator i=constraintSet.begin(), iend=constraintSet.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"BehaviorModel: ";
    for ( Sequence<BehaviorModel>::iterator i=behaviorModel.begin(), iend=behaviorModel.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"VisualModel: ";
    for ( Sequence<VisualModel>::iterator i=visualModel.begin(), iend=visualModel.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"CollisionModel: ";
    for ( Sequence<CollisionModel>::iterator i=collisionModel.begin(), iend=collisionModel.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"ContextObject: ";
    for ( Sequence<ContextObject>::iterator i=contextObject.begin(), iend=contextObject.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"Pipeline: ";
    for ( Single<Pipeline>::iterator i=collisionPipeline.begin(), iend=collisionPipeline.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"VisitorScheduler: ";
    for ( Single<VisitorScheduler>::iterator i=actionScheduler.begin(), iend=actionScheduler.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
}

/** @name Dependency graph
This graph reflects the dependencies between the components. It is used internally to ensure that the initialization order is comform to the dependencies.
*/
/// @{
// Vertices
struct component_t
{
    typedef boost::vertex_property_tag kind;
};
typedef boost::property<component_t, BaseObject*> VertexProperty;

// Graph
typedef ::boost::adjacency_list < ::boost::vecS, ::boost::vecS, ::boost::bidirectionalS, VertexProperty > DependencyGraph;

void Node::sortComponents()
{
    typedef DependencyGraph::vertex_descriptor Vertex;
    DependencyGraph dependencyGraph;
    // map vertex->component
    boost::property_map<DependencyGraph, component_t>::type  component_from_vertex = boost::get( component_t(), dependencyGraph );
    // map component->vertex
    std::map<BaseObject*,Vertex> vertex_from_component;

    // build the graph
    for ( int i=object.size()-1; i>=0; i-- ) // in the reverse order for a final order more similar to the current one
    {
        Vertex v = add_vertex( dependencyGraph );
        component_from_vertex[v] = object[i].get();
        vertex_from_component[object[i].get()] = v;
    }
    assert( depend.getValue().size()%2 == 0 ); // must contain only pairs
    for ( unsigned i=0; i<depend.getValue().size(); i+=2 )
    {
        BaseObject* o1 = getObject( depend.getValue()[i] );
        BaseObject* o2 = getObject( depend.getValue()[i+1] );
        if ( o1==NULL ) cerr<<"Node::sortComponent, could not find object called "<<depend.getValue()[i]<<endl;
        else if ( o2==NULL ) cerr<<"Node::sortComponent, could not find object called "<<depend.getValue()[i+1]<<endl;
        else
        {
            boost::add_edge( vertex_from_component[o1], vertex_from_component[o2], dependencyGraph );
            //cerr<<"Node::sortComponents, added edge "<<o1->getName()<<" -> "<<o2->getName()<<endl;
        }
    }

    // sort the components according to the dependencies
    typedef std::vector< Vertex > container;
    container c;
    boost::topological_sort(dependencyGraph, std::back_inserter(c));

    // remove all the components
    for ( container::reverse_iterator ii=c.rbegin(); ii!=c.rend(); ++ii)
    {
        removeObject(component_from_vertex[*ii]);
    }

    // put the components in the right order
    //cerr << "Node::sortComponents, New component order: ";
    for ( container::reverse_iterator ii=c.rbegin(); ii!=c.rend(); ++ii)
    {
        addObject(component_from_vertex[*ii]);
        //cerr << component_from_vertex[*ii]->getName() << " ";
    }
    //cerr << endl;

}

#ifdef SOFA_SMP
Iterative::IterativePartition* Node::getFirstPartition()
{
    if(is_partition())
        return partition_;
    for (sofa::simulation::Node::ChildIterator it= child.begin(); it != child.end(); ++it)
    {
        sofa::simulation::Node *g=static_cast<sofa::simulation::Node *>(*it);
        if(g)
        {
            Iterative::IterativePartition* p= g->getFirstPartition();
            if(p)
                return p;
        }
    }
    return NULL;
}
#endif


template <class RealObject>
void Node::create( RealObject*& obj, sofa::simulation::xml::Element<sofa::core::objectmodel::BaseNode>*& arg)
{
    obj=(RealObject*)getSimulation()->createNewGraph(arg->getName());
    obj->parse(arg);
}

SOFA_DECL_CLASS(Node)
//create method of Node called if the user wants the default node. The object created will depend on the simulation currently in use.
helper::Creator<xml::NodeElement::Factory, Node> NodeClass("default");

}

}
