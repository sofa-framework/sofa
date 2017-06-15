/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Interface: Node
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_CORE_NODE_H
#define SOFA_SIMULATION_CORE_NODE_H

#include <sofa/core/ExecParams.h>
#include <sofa/core/objectmodel/Context.h>
// moved from GNode (27/04/08)
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/Shader.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseTopologyObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/simulation/simulationcore.h>
#include <sofa/simulation/MutationListener.h>
#include <sofa/simulation/VisitorScheduler.h>

namespace sofa
{
namespace simulation
{
class Visitor;
}
}

#include <sofa/helper/system/thread/CTime.h>
#include <string>
#include <stack>

namespace sofa
{

namespace core
{
namespace visual
{
class VisualParams;
} // namespace visual
} // namespace core

namespace simulation
{

/**
   Implements the object (component) management of the core::Context.
   Contains objects in lists and provides accessors.
   The other nodes are not visible (unknown scene graph).

   @author The SOFA team </www.sofa-framework.org>
 */
class SOFA_SIMULATION_CORE_API Node : public sofa::core::objectmodel::BaseNode, public sofa::core::objectmodel::Context
{

public:
    SOFA_ABSTRACT_CLASS2(Node, BaseNode, Context);

    typedef sofa::core::visual::DisplayFlags DisplayFlags;
protected:
    Node(const std::string& name="");

    virtual ~Node();
public:
    /// Create, add, then return the new child of this Node
    virtual Node::SPtr createChild(const std::string& nodeName)=0;

    /// @name High-level interface
    /// @{

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse ( sofa::core::objectmodel::BaseObjectDescription* arg );

    /// Initialize the components
    void init(const sofa::core::ExecParams* params);
    bool isInitialized() {return initialized;}
    /// Apply modifications to the components
    void reinit(const sofa::core::ExecParams* params);
    /// Do one step forward in time
//    void animate(const core::ExecParams* params, SReal dt);
    /// Draw the objects (using visual visitors)
    void draw(sofa::core::visual::VisualParams* params);
    /// @}

    /// @name Visitor handling
    /// @param precomputedOrder is not used by default but could allow optimization on certain Node specializations
    /// @warning when calling with precomputedOrder=true, the fonction "precomputeTraversalOrder" must be called before executing the visitor and the user must ensure by himself that the simulation graph has done been modified since the last call to "precomputeTraversalOrder"
    /// @{

    /// Execute a recursive action starting from this node.
    virtual void doExecuteVisitor(Visitor* action, bool precomputedOrder=false)=0;

    /// Execute a recursive action starting from this node
    void executeVisitor( simulation::Visitor* action, bool precomputedOrder=false);

    /// Execute a recursive action starting from this node
    void execute(simulation::Visitor& action, bool precomputedOrder=false)
    {
        simulation::Visitor* p = &action;
        executeVisitor(p, precomputedOrder);
    }

    /// Execute a recursive action starting from this node
    void execute(simulation::Visitor* p, bool precomputedOrder=false)
    {
        executeVisitor(p, precomputedOrder);
    }

    /// Execute a recursive action starting from this node
    template<class Act, class Params>
    void execute(const Params* params, bool precomputedOrder=false)
    {
        Act action(params);
        simulation::Visitor* p = &action;
        executeVisitor(p, precomputedOrder);
    }

    /// Execute a recursive action starting from this node
    template<class Act>
    void execute(core::visual::VisualParams* vparams, bool precomputedOrder=false)
    {
        Act action(vparams);
        simulation::Visitor* p = &action;
        executeVisitor(p, precomputedOrder);
    }

    /// Possible optimization with traversal precomputation, not mandatory and does nothing by default
    virtual void precomputeTraversalOrder( const sofa::core::ExecParams* ) {}

    /// @}

    /// @name Component containers
    /// @{
    // methods moved from GNode (27/04/08)

    /// Sequence class to hold a list of objects. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end).
    /// UPDATE: it is now an alias for the Link pointer container
    template < class T, bool strong = false >
    class Sequence : public MultiLink<Node, T, BaseLink::FLAG_DOUBLELINK|(strong ? BaseLink::FLAG_STRONGLINK : BaseLink::FLAG_DUPLICATE)>
    {
    public:
        typedef MultiLink<Node, T, BaseLink::FLAG_DOUBLELINK|(strong ? BaseLink::FLAG_STRONGLINK : BaseLink::FLAG_DUPLICATE)> Inherit;
        typedef T pointed_type;
        typedef typename Inherit::DestPtr value_type;
        //typedef TPtr value_type;
        typedef typename Inherit::const_iterator const_iterator;
        typedef typename Inherit::const_reverse_iterator const_reverse_iterator;
        typedef const_iterator iterator;
        typedef const_reverse_iterator reverse_iterator;

        Sequence(const BaseLink::InitLink<Node>& init)
            : Inherit(init)
        {
        }

        value_type operator[](unsigned int i) const
        {
            return this->get(i);
        }

        /// Swap two values in the list. Uses a const_cast to violate the read-only iterators.
        void swap( iterator a, iterator b )
        {
            value_type& wa = const_cast<value_type&>(*a);
            value_type& wb = const_cast<value_type&>(*b);
            value_type tmp = *a;
            wa = *b;
            wb = tmp;
        }
    };

    /// Class to hold 0-or-1 object. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end), plus an automatic convertion to one pointer.
    /// UPDATE: it is now an alias for the Link pointer container
    template < class T, bool duplicate = true >
    class Single : public SingleLink<Node, T, BaseLink::FLAG_DOUBLELINK|(duplicate ? BaseLink::FLAG_DUPLICATE : BaseLink::FLAG_NONE)>
    {
    public:
        typedef SingleLink<Node, T, BaseLink::FLAG_DOUBLELINK|(duplicate ? BaseLink::FLAG_DUPLICATE : BaseLink::FLAG_NONE)> Inherit;
        typedef T pointed_type;
        typedef typename Inherit::DestPtr value_type;
        //typedef TPtr value_type;
        typedef typename Inherit::const_iterator const_iterator;
        typedef typename Inherit::const_reverse_iterator const_reverse_iterator;
        typedef const_iterator iterator;
        typedef const_reverse_iterator reverse_iterator;

        Single(const BaseLink::InitLink<Node>& init)
            : Inherit(init)
        {
        }

        T* operator->() const
        {
            return this->get();
        }
        T& operator*() const
        {
            return *this->get();
        }
        operator T*() const
        {
            return this->get();
        }
    };

    Sequence<Node,true> child;
    typedef Sequence<Node,true>::iterator ChildIterator;

    Sequence<sofa::core::objectmodel::BaseObject,true> object;
    typedef Sequence<sofa::core::objectmodel::BaseObject,true>::iterator ObjectIterator;
    typedef Sequence<sofa::core::objectmodel::BaseObject,true>::reverse_iterator ObjectReverseIterator;

    Single<sofa::core::behavior::BaseAnimationLoop> animationManager;
    Single<sofa::core::visual::VisualLoop> visualLoop;

    Sequence<sofa::core::BehaviorModel> behaviorModel;
    Sequence<sofa::core::BaseMapping> mapping;

    Sequence<sofa::core::behavior::OdeSolver> solver;
    Sequence<sofa::core::behavior::ConstraintSolver> constraintSolver;
    Sequence<sofa::core::behavior::BaseLinearSolver> linearSolver;

    Single<sofa::core::topology::Topology> topology;
    Single<sofa::core::topology::BaseMeshTopology> meshTopology;
    Sequence<sofa::core::topology::BaseTopologyObject> topologyObject;

    Single<sofa::core::BaseState> state;
    Single<sofa::core::behavior::BaseMechanicalState> mechanicalState;
    Single<sofa::core::BaseMapping> mechanicalMapping;
    Single<sofa::core::behavior::BaseMass> mass;
    Sequence<sofa::core::behavior::BaseForceField> forceField;
    Sequence<sofa::core::behavior::BaseInteractionForceField> interactionForceField;
    Sequence<sofa::core::behavior::BaseProjectiveConstraintSet> projectiveConstraintSet;
    Sequence<sofa::core::behavior::BaseConstraintSet> constraintSet;
    Sequence<sofa::core::objectmodel::ContextObject> contextObject;
    Sequence<sofa::core::objectmodel::ConfigurationSetting> configurationSetting;

    Sequence<sofa::core::visual::Shader> shaders;
    Sequence<sofa::core::visual::VisualModel> visualModel;
    Sequence<sofa::core::visual::VisualManager> visualManager;

    Sequence<sofa::core::CollisionModel> collisionModel;
    Single<sofa::core::collision::Pipeline> collisionPipeline;

    Sequence<sofa::core::objectmodel::BaseObject> unsorted;

    /// @}


    /// @name Set/get objects
    /// @{

    /// Add an object and return this. Detect the implemented interfaces and add the object to the corresponding lists.
    virtual bool addObject(sofa::core::objectmodel::BaseObject::SPtr obj);

    /// Remove an object
    virtual bool removeObject(sofa::core::objectmodel::BaseObject::SPtr obj);

    /// Move an object from another node
    virtual void moveObject(sofa::core::objectmodel::BaseObject::SPtr obj);

    /// Find an object given its name
    sofa::core::objectmodel::BaseObject* getObject(const std::string& name) const;

    void* findLinkDestClass(const sofa::core::objectmodel::BaseClass* destType, const std::string& path, const sofa::core::objectmodel::BaseLink* link);

#ifdef SOFA_SMP
    /// Get first partition
    Iterative::IterativePartition* getFirstPartition();
#endif

    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const=0;

    /// Generic object access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, SearchDirection dir = SearchUp) const
    {
        return getObject(class_info, sofa::core::objectmodel::TagSet(), dir);
    }

    /// Generic object access, given a path from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const=0;

    /// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const =0;

    /// Generic list of objects access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const
    {
        getObjects(class_info, container, sofa::core::objectmodel::TagSet(), dir);
    }




    /// List all objects of this node deriving from a given class
    template<class Object, class Container>
    void getNodeObjects(Container* list)
    {
        this->get<Object, Container>(list, Local);
    }

    /// Return an object of this node deriving from a given class, or NULL if not found.
    /// Note that only the first object is returned.
    template<class Object>
    void getNodeObject(Object*& result)
    {
        result = this->get<Object>(Local);
    }

    template<class Object>
    Object* getNodeObject()
    {
        return this->get<Object>(Local);
    }

    /// List all objects of this node and sub-nodes deriving from a given class
    template<class Object, class Container>
    void getTreeObjects(Container* list)
    {
        this->get<Object, Container>(list, SearchDown);
    }

    /// Return an object of this node and sub-nodes deriving from a given class, or NULL if not found.
    /// Note that only the first object is returned.
    template<class Object>
    void getTreeObject(Object*& result)
    {
        result = this->get<Object>(SearchDown);
    }

    template<class Object>
    Object* getTreeObject()
    {
        return this->get<Object>(SearchDown);
    }

    /// Topology
    virtual sofa::core::topology::Topology* getTopology() const;

    /// Mesh Topology (unified interface for both static and dynamic topologies)
    virtual sofa::core::topology::BaseMeshTopology* getMeshTopology() const;

    /// Degrees-of-Freedom
    virtual sofa::core::BaseState* getState() const;

    /// Mechanical Degrees-of-Freedom
    virtual sofa::core::behavior::BaseMechanicalState* getMechanicalState() const;

    /// Shader
    virtual sofa::core::visual::Shader* getShader() const;
    virtual sofa::core::visual::Shader* getShader(const sofa::core::objectmodel::TagSet& t) const;

    /// @name Solvers and main algorithms
    /// @{

    virtual sofa::core::behavior::BaseAnimationLoop* getAnimationLoop() const;
    virtual sofa::core::behavior::OdeSolver* getOdeSolver() const;
    virtual sofa::core::collision::Pipeline* getCollisionPipeline() const;
    virtual sofa::core::visual::VisualLoop* getVisualLoop() const;

    /// @}

    /// Remove odesolvers and mastercontroler
    virtual void removeControllers();

    /// Find a child node given its name
    Node* getChild(const std::string& name) const;

    /// Get a descendant node given its name
    Node* getTreeNode(const std::string& name) const;

    /// Get children nodes
    virtual Children getChildren() const;

    BaseContext* getRootContext() const
    {
        return getRoot()->getContext();
    }

    Node* setDebug(bool);
    bool getDebug() const;
    // debug
    void printComponents();

    const BaseContext* getContext() const;
    BaseContext* getContext();

    /// Update the whole context values, based on parent and local ContextObjects
    virtual void updateContext();

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    virtual void updateSimulationContext();

    /// Called during initialization to corectly propagate the visual context to the children
    virtual void initVisualContext() {}

    /// Propagate an event
    virtual void propagateEvent(const sofa::core::ExecParams* params, sofa::core::objectmodel::Event* event);

    /// Update the visual context values, based on parent and local ContextObjects
    virtual void updateVisualContext();

    // VisitorScheduler can use doExecuteVisitor() method
    friend class VisitorScheduler;

    /// Must be called after each graph modification. Do not call it directly, apply an InitVisitor instead.
    virtual void initialize();

    virtual void bwdInit();

    /// Called after initialization to set the default value of the visual context.
    virtual void setDefaultVisualContextValue();

    template <class RealObject>
    static Node::SPtr create(RealObject*, core::objectmodel::BaseObjectDescription* arg);


    static Node::SPtr create( const std::string& name );

    /// return the smallest common parent between this and node2 (returns NULL if separated sub-graphes)
    virtual Node* findCommonParent( simulation::Node* node2 ) = 0;

	/// override context setSleeping to add notification.
	virtual void setSleeping(bool /*val*/);

protected:
    bool debug_;
    bool initialized;

    virtual void doAddObject(sofa::core::objectmodel::BaseObject::SPtr obj);
    virtual void doRemoveObject(sofa::core::objectmodel::BaseObject::SPtr obj);


    std::stack<Visitor*> actionStack;

    virtual void notifyAddChild(Node::SPtr node);
    virtual void notifyRemoveChild(Node::SPtr node);
    virtual void notifyMoveChild(Node::SPtr node, Node* prev);
    virtual void notifyAddObject(sofa::core::objectmodel::BaseObject::SPtr obj);
    virtual void notifyRemoveObject(sofa::core::objectmodel::BaseObject::SPtr obj);
    virtual void notifyMoveObject(sofa::core::objectmodel::BaseObject::SPtr obj, Node* prev);
	virtual void notifySleepChanged();


    BaseContext* _context;

    helper::vector<MutationListener*> listener;


public:

    virtual void notifyAddSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave);
    virtual void notifyRemoveSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave);
    virtual void notifyMoveSlave(sofa::core::objectmodel::BaseObject* previousMaster, sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave);

    virtual void addListener(MutationListener* obj);
    virtual void removeListener(MutationListener* obj);

    // Added by FF to model component dependencies
    /// Pairs representing component dependencies. First must be initialized before second.
    Data < sofa::helper::vector < std::string > > depend;

    /// Sort the components according to the dependencies expressed in Data depend.
    void sortComponents();



    /// @name virtual functions to add/remove some special components direclty in the right Sequence
    /// @{

#define NODE_ADD_IN_SEQUENCE( CLASSNAME, FUNCTIONNAME, SEQUENCENAME ) \
    virtual void add##FUNCTIONNAME( CLASSNAME* obj ) { SEQUENCENAME.add(obj); } \
    virtual void remove##FUNCTIONNAME( CLASSNAME* obj ) { SEQUENCENAME.remove(obj); }

    // WARNINGS subtilities:
    // an InteractioFF is NOT in the FF Sequence
    // a MechanicalMapping is NOT in the Mapping Sequence
    // a Mass is in the FF Sequence
    // a MeshTopology is in the topology Sequence

public:

    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseAnimationLoop, AnimationLoop, animationManager )
    NODE_ADD_IN_SEQUENCE( sofa::core::visual::VisualLoop, VisualLoop, visualLoop )
    NODE_ADD_IN_SEQUENCE( sofa::core::BehaviorModel, BehaviorModel, behaviorModel )
    NODE_ADD_IN_SEQUENCE( sofa::core::BaseMapping, Mapping, mapping )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::OdeSolver, OdeSolver, solver )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::ConstraintSolver, ConstraintSolver, constraintSolver )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseLinearSolver, LinearSolver, linearSolver )
    NODE_ADD_IN_SEQUENCE( sofa::core::topology::Topology, Topology, topology )
    NODE_ADD_IN_SEQUENCE( sofa::core::topology::BaseMeshTopology, MeshTopology, meshTopology )
    NODE_ADD_IN_SEQUENCE( sofa::core::topology::BaseTopologyObject, TopologyObject, topologyObject )
    NODE_ADD_IN_SEQUENCE( sofa::core::BaseState, State, state )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseMechanicalState,MechanicalState, mechanicalState )
    NODE_ADD_IN_SEQUENCE( sofa::core::BaseMapping, MechanicalMapping, mechanicalMapping )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseMass, Mass, mass )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseForceField, ForceField, forceField )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseInteractionForceField, InteractionForceField, interactionForceField )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseProjectiveConstraintSet, ProjectiveConstraintSet, projectiveConstraintSet )
    NODE_ADD_IN_SEQUENCE( sofa::core::behavior::BaseConstraintSet, ConstraintSet, constraintSet )
    NODE_ADD_IN_SEQUENCE( sofa::core::objectmodel::ContextObject, ContextObject, contextObject )
    NODE_ADD_IN_SEQUENCE( sofa::core::objectmodel::ConfigurationSetting, ConfigurationSetting, configurationSetting )
    NODE_ADD_IN_SEQUENCE( sofa::core::visual::Shader, Shader, shaders )
    NODE_ADD_IN_SEQUENCE( sofa::core::visual::VisualModel, VisualModel, visualModel )
    NODE_ADD_IN_SEQUENCE( sofa::core::visual::VisualManager, VisualManager, visualManager )
    NODE_ADD_IN_SEQUENCE( sofa::core::CollisionModel, CollisionModel, collisionModel )
    NODE_ADD_IN_SEQUENCE( sofa::core::collision::Pipeline, CollisionPipeline, collisionPipeline )

#undef NODE_ADD_IN_SEQUENCE

    /// @}

};

}

}

#endif
