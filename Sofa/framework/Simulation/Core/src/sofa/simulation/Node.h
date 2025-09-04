/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/simulation/config.h>
#include <sofa/simulation/fwd.h>

#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/simulation/Visitor.h>

#include <type_traits>
#include <string>
#include <stack>

namespace sofa::simulation
{

/// @name Component containers
/// @{
/// Sequence class to hold a list of objects. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end).
/// UPDATE: it is now an alias for the Link pointer container
template < class T, bool strong = false >
class NodeSequence : public MultiLink<Node, T, BaseLink::FLAG_DOUBLELINK|(strong ? BaseLink::FLAG_STRONGLINK : BaseLink::FLAG_DUPLICATE)>
{
public:
    typedef MultiLink<Node, T, BaseLink::FLAG_DOUBLELINK|(strong ? BaseLink::FLAG_STRONGLINK : BaseLink::FLAG_DUPLICATE)> Inherit;
    typedef T pointed_type;
    typedef typename Inherit::DestPtr value_type;
    typedef typename Inherit::const_iterator const_iterator;
    typedef typename Inherit::const_reverse_iterator const_reverse_iterator;
    typedef const_iterator iterator;
    typedef const_reverse_iterator reverse_iterator;

    NodeSequence(const BaseLink::InitLink<Node>& init)
        : Inherit(init)
    {
    }

    value_type operator[](std::size_t i) const
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

/// Class to hold 0-or-1 object. Public access is only readonly using an interface similar to std::vector (size/[]/begin/end), plus an automatic conversion to one pointer.
/// UPDATE: it is now an alias for the Link pointer container
template < class T, bool duplicate = true >
class NodeSingle : public SingleLink<Node, T, BaseLink::FLAG_DOUBLELINK|(duplicate ? BaseLink::FLAG_DUPLICATE : BaseLink::FLAG_NONE)>
{
public:
    typedef SingleLink<Node, T, BaseLink::FLAG_DOUBLELINK|(duplicate ? BaseLink::FLAG_DUPLICATE : BaseLink::FLAG_NONE)> Inherit;
    typedef T pointed_type;
    typedef typename Inherit::DestPtr value_type;
    typedef typename Inherit::const_iterator const_iterator;
    typedef typename Inherit::const_reverse_iterator const_reverse_iterator;
    typedef const_iterator iterator;
    typedef const_reverse_iterator reverse_iterator;

    NodeSingle(const BaseLink::InitLink<Node>& init)
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


extern template class NodeSequence<Node,true>;
extern template class NodeSequence<sofa::core::objectmodel::BaseObject,true>;
extern template class NodeSequence<sofa::core::BehaviorModel>;
extern template class NodeSequence<sofa::core::BaseMapping>;
extern template class NodeSequence<sofa::core::behavior::OdeSolver>;
extern template class NodeSequence<sofa::core::behavior::ConstraintSolver>;
extern template class NodeSequence<sofa::core::behavior::BaseLinearSolver>;
extern template class NodeSequence<sofa::core::topology::BaseTopologyObject>;
extern template class NodeSequence<sofa::core::behavior::BaseForceField>;
extern template class NodeSequence<sofa::core::behavior::BaseInteractionForceField>;
extern template class NodeSequence<sofa::core::behavior::BaseProjectiveConstraintSet>;
extern template class NodeSequence<sofa::core::behavior::BaseConstraintSet>;
extern template class NodeSequence<sofa::core::objectmodel::ContextObject>;
extern template class NodeSequence<sofa::core::objectmodel::ConfigurationSetting>;
extern template class NodeSequence<sofa::core::visual::Shader>;
extern template class NodeSequence<sofa::core::visual::VisualModel>;
extern template class NodeSequence<sofa::core::visual::VisualManager>;
extern template class NodeSequence<sofa::core::CollisionModel>;
extern template class NodeSequence<sofa::core::objectmodel::BaseObject>;

extern template class NodeSingle<sofa::core::behavior::BaseAnimationLoop>;
extern template class NodeSingle<sofa::core::visual::VisualLoop>;
extern template class NodeSingle<sofa::core::visual::BaseVisualStyle>;
extern template class NodeSingle<sofa::core::topology::Topology>;
extern template class NodeSingle<sofa::core::topology::BaseMeshTopology>;
extern template class NodeSingle<sofa::core::BaseState>;
extern template class NodeSingle<sofa::core::behavior::BaseMechanicalState>;
extern template class NodeSingle<sofa::core::BaseMapping>;
extern template class NodeSingle<sofa::core::behavior::BaseMass>;
extern template class NodeSingle<sofa::core::collision::Pipeline>;


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

    Node(const std::string& name="", Node* parent=nullptr);
    virtual ~Node() override;

    /// @name High-level interface
    /// @{

    /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

    /// Initialize the components
    void init(const sofa::core::ExecParams* params);
    bool isInitialized() const {return initialized;}
    /// Apply modifications to the components
    void reinit(const sofa::core::ExecParams* params);
    /// Draw the objects (using visual visitors)
    void draw(sofa::core::visual::VisualParams* params);
    /// @}

    /// @name Visitor handling
    /// @param precomputedOrder is not used by default but could allow optimization on certain Node specializations
    /// @warning when calling with precomputedOrder=true, the function "precomputeTraversalOrder" must be called before executing the visitor and the user must ensure by himself that the simulation graph has done been modified since the last call to "precomputeTraversalOrder"
    /// @{

    /// Execute a recursive action starting from this node
    void executeVisitor(Visitor* action, bool precomputedOrder=false) override;

    /// Execute a recursive action starting from this node
    void execute(Visitor& action, bool precomputedOrder=false)
    {
        simulation::Visitor* p = &action;
        executeVisitor(p, precomputedOrder);
    }

    /// Execute a recursive action starting from this node
    void execute(Visitor* p, bool precomputedOrder=false)
    {
        executeVisitor(p, precomputedOrder);
    }

    /// Execute a recursive action starting from this node
    template<class Act, class Params>
    void execute(const Params* params, bool precomputedOrder=false)
    {
        Act action(params);
        Visitor* p = &action;
        executeVisitor(p, precomputedOrder);
    }

    /// Execute a recursive action starting from this node
    template<class Act>
    void execute(sofa::core::visual::VisualParams* vparams, bool precomputedOrder=false)
    {
        Act action(vparams);
        Visitor* p = &action;
        executeVisitor(p, precomputedOrder);
    }

    /// Possible optimization with traversal precomputation, not mandatory and does nothing by default
    void precomputeTraversalOrder( const sofa::core::ExecParams* );

    /// @}

    template<class A, bool B=false>
    using Sequence = NodeSequence<A,B>;

    template<class A, bool B=true>
    using Single = NodeSingle<A,B>;


    NodeSequence<Node,true> child;
    typedef NodeSequence<Node,true>::iterator ChildIterator;

    NodeSequence<sofa::core::objectmodel::BaseObject,true> object;
    typedef NodeSequence<sofa::core::objectmodel::BaseObject,true>::iterator ObjectIterator;
    typedef NodeSequence<sofa::core::objectmodel::BaseObject,true>::reverse_iterator ObjectReverseIterator;

    NodeSequence<sofa::core::BehaviorModel> behaviorModel;
    NodeSequence<sofa::core::BaseMapping> mapping;

    NodeSequence<sofa::core::behavior::OdeSolver> solver;
    NodeSequence<sofa::core::behavior::ConstraintSolver> constraintSolver;
    NodeSequence<sofa::core::behavior::BaseLinearSolver> linearSolver;
    NodeSequence<sofa::core::topology::BaseTopologyObject> topologyObject;
    NodeSequence<sofa::core::behavior::BaseForceField> forceField;
    NodeSequence<sofa::core::behavior::BaseInteractionForceField> interactionForceField;
    NodeSequence<sofa::core::behavior::BaseProjectiveConstraintSet> projectiveConstraintSet;
    NodeSequence<sofa::core::behavior::BaseConstraintSet> constraintSet;
    NodeSequence<sofa::core::objectmodel::ContextObject> contextObject;
    NodeSequence<sofa::core::objectmodel::ConfigurationSetting> configurationSetting;
    NodeSequence<sofa::core::visual::Shader> shaders;
    NodeSequence<sofa::core::visual::VisualModel> visualModel;
    NodeSequence<sofa::core::visual::VisualManager> visualManager;
    NodeSequence<sofa::core::CollisionModel> collisionModel;
    NodeSequence<sofa::core::objectmodel::BaseObject> unsorted;

    NodeSingle<sofa::core::behavior::BaseAnimationLoop> animationManager;
    NodeSingle<sofa::core::visual::VisualLoop> visualLoop;
    NodeSingle<sofa::core::visual::BaseVisualStyle> visualStyle;
    NodeSingle<sofa::core::topology::Topology> topology;
    NodeSingle<sofa::core::topology::BaseMeshTopology> meshTopology;
    NodeSingle<sofa::core::BaseState> state;
    NodeSingle<sofa::core::behavior::BaseMechanicalState> mechanicalState;
    NodeSingle<sofa::core::BaseMapping> mechanicalMapping;
    NodeSingle<sofa::core::behavior::BaseMass> mass;
    NodeSingle<sofa::core::collision::Pipeline> collisionPipeline;
    /// @}

    /// @name Set/get objects
    /// @{

    /// Pure Virtual method from BaseNode
    /// Add a child node
    virtual void addChild(BaseNode::SPtr node) final;
    /// Remove a child node
    virtual void removeChild(BaseNode::SPtr node) final;
    /// Move a node in this from another node
    virtual void moveChild(BaseNode::SPtr node, BaseNode::SPtr prev_parent) final;

    /// @name Set/get objects
    /// @{

    /// Add an object and return this. Detect the implemented interfaces and add the object to the corresponding lists.
    virtual bool addObject(sofa::core::objectmodel::BaseObject::SPtr obj, sofa::core::objectmodel::TypeOfInsertion insertionLocation=sofa::core::objectmodel::TypeOfInsertion::AtEnd) final;

    /// Remove an object
    virtual bool removeObject(sofa::core::objectmodel::BaseObject::SPtr obj) final;

    /// Move an object from another node
    virtual void moveObject(sofa::core::objectmodel::BaseObject::SPtr obj) final;

    /// Find an object given its name
    sofa::core::objectmodel::BaseObject* getObject(const std::string& name) const;

    Base* findLinkDestClass(const sofa::core::objectmodel::BaseClass* destType, const std::string& path, const sofa::core::objectmodel::BaseLink* link) override;

    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///


    /// Note that the template wrapper method should generally be used to have the correct return type,
    void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, SearchDirection dir = SearchUp) const override
    {
        return getObject(class_info, sofa::core::objectmodel::TagSet(), dir);
    }

    /// Generic list of objects access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const override
    {
        getObjects(class_info, container, sofa::core::objectmodel::TagSet(), dir);
    }

    /// get node's local objects respecting specified class_info and tags
    void getLocalObjects( const sofa::core::objectmodel::ClassInfo& class_info, Node::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags ) const ;

    /// List all objects of this node deriving from a given class
    template<class Object, class Container>
    void getNodeObjects(Container* list)
    {
       return BaseContext::getObjects<Object, Container>(list, Local) ;
    }

    /// Returns a list of object of type passed as a parameter.
    template<class Container>
    Container* getNodeObjects(Container* result)
    {
        return BaseContext::getObjects(result, Local) ;
    }

    /// Returns a list of object of type passed as a parameter
    template<class Container>
    Container& getNodeObjects(Container& result)
    {
        return BaseContext::getObjects(result, Local);
    }

    /// Returns a list of object of type passed as a parameter.
    /// This function is return object by copy but should be compatible with
    /// Return Value Optimization so the copy should be removed by the compiler.
    /// Eg:
    ///     for( BaseObject* o : node->getNodeObjects() ) { ... }
    ///     for( VisualModel* v : node->getNodeObjects<VisualModel>() ) { ... }
    template<class Object=sofa::core::objectmodel::BaseObject>
    std::vector<Object*> getNodeObjects()
    {
        std::vector<Object*> tmp ;
        BaseContext::getObjects(tmp, Local);
        return tmp;
    }

    /// Return an object of this node deriving from a given class, or nullptr if not found.
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

    /// List all objects of this node and sub-nodes deriving from a given class
    template<class Container>
    Container* getTreeObjects(Container* result)
    {
        return BaseContext::getObjects(result,  SearchDown);
    }

    /// List all objects of this node and sub-nodes deriving from a given class
    template<class Container>
    Container& getTreeObjects(Container& result)
    {
        return BaseContext::getObjects(result,  SearchDown);
    }

    /// List all objects of this node and sub-nodes deriving from a given class
    /// This function is return a std::vector by copy but should be compatible with
    /// Return Value Optimization so the copy should be removed by the compiler.
    /// Eg:
    ///     for( BaseObject* o : node->getTreeObjects() ) { ... }
    ///     for( VisualModel* v : node->getTreeObjects<VisualModel>() ) { ... }
    template<class Object=sofa::core::objectmodel::BaseObject>
    std::vector<Object*> getTreeObjects()
    {
        std::vector<Object*> tmp ;
        BaseContext::getObjects(tmp, SearchDown);
        return tmp;
    }



    /// Return an object of this node and sub-nodes deriving from a given class, or nullptr if not found.
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
    sofa::core::topology::Topology* getTopology() const override;

    /// Degrees-of-Freedom
    sofa::core::BaseState* getState() const override;

    /// Mechanical Degrees-of-Freedom
    sofa::core::behavior::BaseMechanicalState* getMechanicalState() const override;

    /// Shader
    sofa::core::visual::Shader* getShader() const override;
    virtual sofa::core::visual::Shader* getShader(const sofa::core::objectmodel::TagSet& t) const;

    /// @name Solvers and main algorithms
    /// @{

    sofa::core::behavior::BaseAnimationLoop* getAnimationLoop() const override;
    sofa::core::behavior::OdeSolver* getOdeSolver() const override;
    sofa::core::collision::Pipeline* getCollisionPipeline() const override;
    sofa::core::visual::VisualLoop* getVisualLoop() const override;

    /// @}

    /// Remove odesolvers and mastercontroler
    virtual void removeControllers();

    /// Find a child node given its name
    Node* getChild(const std::string& name) const;

    /// Get a descendant node given its name
    Node* getTreeNode(const std::string& name) const;
    /// Get a node in the scene graph matching the given absolute path
    Node* getNodeInGraph(const std::string& absolutePath) const;

    /// Get children nodes
    Children getChildren() const override;

    BaseContext* getRootContext() const override
    {
        return getRoot()->getContext();
    }

    Node* setDebug(bool);
    bool getDebug() const;
    void printComponents();

    const BaseContext* getContext() const override;
    BaseContext* getContext() override;

    /// Update the whole context values, based on parent and local ContextObjects
    virtual void updateContext();

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    virtual void updateSimulationContext();

    /// Called during initialization to correctly propagate the visual context to the children
    virtual void initVisualContext();

    /// Propagate an event
    void propagateEvent(const sofa::core::ExecParams* params, sofa::core::objectmodel::Event* event) override;

    /// Update the visual context values, based on parent and local ContextObjects
    virtual void updateVisualContext();

    // VisitorScheduler can use doExecuteVisitor() method
    friend class VisitorScheduler;

    /// Must be called after each graph modification. Do not call it directly, apply an InitVisitor instead.
    virtual void initialize();

    /// Called after initialization to set the default value of the visual context.
    virtual void setDefaultVisualContextValue();

    template <class RealObject>
    static Node::SPtr create(RealObject*, sofa::core::objectmodel::BaseObjectDescription* arg);

    /// override context setSleeping to add notification.
    void setSleeping(bool val) override;

public:
    virtual void addListener(MutationListener* obj);
    virtual void removeListener(MutationListener* obj);

    static const std::string GetCustomClassName(){ return "Node"; }

    Node::SPtr createChild(const std::string& nodeName);

    /// Remove the current node from the graph: consists in removing the link to its parent
    void detachFromGraph() override;

    /// Get a list of parent node
    Parents getParents() const override;

    /// returns number of parents
    size_t getNbParents() const override;

    /// return the first parent (returns nullptr if no parent)
    BaseNode* getFirstParent() const override;

    /// Test if the given node is a parent of this node.
    bool hasParent(const BaseNode* node) const override;

    /// Test if the given context is a parent of this context.
    bool hasParent(const BaseContext* context) const;

    /// Test if the given context is an ancestor of this context.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    bool hasAncestor(const BaseNode* node) const override
    {
        return hasAncestor(node->getContext());
    }

    /// Test if the given context is an ancestor of this context.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    bool hasAncestor(const BaseContext* context) const override;


    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const override;

    /// Generic object access, given a path from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const override;

    /// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const override;

    /// Mesh Topology that is relevant for this context
    /// (within it or its parents until a mapping is reached that does not preserve topologies).
    sofa::core::topology::BaseMeshTopology* getMeshTopologyLink(SearchDirection dir = SearchUp) const override;

    static Node::SPtr create(Node*, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        Node::SPtr obj = Node::SPtr(new Node());
        obj->parse(arg);
        return obj;
    }

    void moveChild(BaseNode::SPtr node) override;

    /// return the smallest common parent between this and node2 (returns nullptr if separated sub-graphes)
    Node* findCommonParent( simulation::Node* node2 );

protected:
    bool debug_;
    bool initialized;

    virtual bool doAddObject(sofa::core::objectmodel::BaseObject::SPtr obj,  sofa::core::objectmodel::TypeOfInsertion insertionLocation= sofa::core::objectmodel::TypeOfInsertion::AtEnd);
    virtual bool doRemoveObject(sofa::core::objectmodel::BaseObject::SPtr obj);
    virtual void doMoveObject(sofa::core::objectmodel::BaseObject::SPtr sobj, Node* prev_parent);

    std::stack<Visitor*> actionStack;

private:    
    virtual void notifyBeginAddChild(Node::SPtr parent, Node::SPtr child) const;
    virtual void notifyBeginRemoveChild(Node::SPtr parent, Node::SPtr child) const;

    virtual void notifyBeginAddObject(Node::SPtr parent, sofa::core::objectmodel::BaseObject::SPtr obj) const;
    virtual void notifyBeginRemoveObject(Node::SPtr parent, sofa::core::objectmodel::BaseObject::SPtr obj) const;

    virtual void notifyEndAddChild(Node::SPtr parent, Node::SPtr child) const;
    virtual void notifyEndRemoveChild(Node::SPtr parent, Node::SPtr child) const;

    virtual void notifyEndAddObject(Node::SPtr parent, sofa::core::objectmodel::BaseObject::SPtr obj) const;
    virtual void notifyEndRemoveObject(Node::SPtr parent, sofa::core::objectmodel::BaseObject::SPtr obj) const;

    virtual void notifySleepChanged(Node* node) const;

    virtual void notifyBeginAddSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave) const;
    virtual void notifyBeginRemoveSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave) const;

    virtual void notifyEndAddSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave) const;
    virtual void notifyEndRemoveSlave(sofa::core::objectmodel::BaseObject* master, sofa::core::objectmodel::BaseObject* slave) const;

    // init all contextObject.
    void initializeContexts();
protected:
    BaseContext* _context;

    type::vector<MutationListener*> listener;

    /// @name virtual functions to add/remove some special components directly in the right Sequence
    /// @{

#define NODE_DECLARE_SEQUENCE_ACCESSOR( CLASSNAME, FUNCTIONNAME, SEQUENCENAME ) \
    void add##FUNCTIONNAME( CLASSNAME* obj ) override ; \
    void remove##FUNCTIONNAME( CLASSNAME* obj ) override ;

    /// WARNINGS subtilities:
    /// an InteractioFF is NOT in the FF Sequence
    /// a MechanicalMapping is NOT in the Mapping Sequence
    /// a Mass is in the FF Sequence
    /// a MeshTopology is in the topology Sequence
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseAnimationLoop, AnimationLoop, animationManager )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::visual::VisualLoop, VisualLoop, visualLoop )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::BehaviorModel, BehaviorModel, behaviorModel )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::BaseMapping, Mapping, mapping )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::OdeSolver, OdeSolver, solver )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::ConstraintSolver, ConstraintSolver, constraintSolver )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseLinearSolver, LinearSolver, linearSolver )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::topology::Topology, Topology, topology )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::topology::BaseMeshTopology, MeshTopology, meshTopology )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::topology::BaseTopologyObject, TopologyObject, topologyObject )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::BaseState, State, state )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseMechanicalState,MechanicalState, mechanicalState )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::BaseMapping, MechanicalMapping, mechanicalMapping )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseMass, Mass, mass )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseForceField, ForceField, forceField )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseInteractionForceField, InteractionForceField, interactionForceField )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseProjectiveConstraintSet, ProjectiveConstraintSet, projectiveConstraintSet )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseConstraintSet, ConstraintSet, constraintSet )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::objectmodel::ContextObject, ContextObject, contextObject )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::objectmodel::ConfigurationSetting, ConfigurationSetting, configurationSetting )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::visual::Shader, Shader, shaders )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::visual::VisualModel, VisualModel, visualModel )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::visual::BaseVisualStyle, VisualStyle, visualStyle )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::visual::VisualManager, VisualManager, visualManager )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::CollisionModel, CollisionModel, collisionModel )
    NODE_DECLARE_SEQUENCE_ACCESSOR( sofa::core::collision::Pipeline, CollisionPipeline, collisionPipeline )

#undef NODE_DECLARE_SEQUENCE_ACCESSOR

    /// @}


    /// FROM DAG NODE

    typedef MultiLink<Node,Node,BaseLink::FLAG_STOREPATH|BaseLink::FLAG_DOUBLELINK> LinkParents;
    typedef LinkParents::const_iterator ParentIterator;

private:
    /// bottom-up traversal, returning the first node which have a descendancy containing both node1 & node2
    Node* findCommonParent( Node* node1, Node* node2 );

    LinkParents l_parents;

    void doAddChild(BaseNode::SPtr node);
    void doRemoveChild(BaseNode::SPtr node);
    void doMoveChild(BaseNode::SPtr node, BaseNode::SPtr previous_parent);

    /// Execute a recursive action starting from this node.
    void doExecuteVisitor(simulation::Visitor* action, bool precomputedOrder=false);

    /// @name @internal stuff related to the DAG traversal
    /// @{

    /// all child nodes (unordered)
    std::set<Node*> _descendancy;

    /// bottom-up traversal removing descendancy
    void setDirtyDescendancy();

    /// traversal updating the descendancy
    void updateDescendancy();

    /// traversal flags
    typedef enum
    {
        NOT_VISITED=0,
        VISITED,
        PRUNED
    } VisitedStatus;

    /// wrapper to use VisitedStatus in a std::map (to ensure the default map insertion will give NOT_VISITED)
    struct StatusStruct
    {
        StatusStruct() : status(NOT_VISITED) {}
        StatusStruct( const VisitedStatus& s ) : status(s) {}
        inline void operator=( const VisitedStatus& s ) { status=s; }
        inline bool operator==( const VisitedStatus& s ) const { return status==s; }
        inline bool operator==( const StatusStruct& s ) const { return status==s.status; }
        inline bool operator!=( const VisitedStatus& s ) const { return status!=s; }
        inline bool operator!=( const StatusStruct& s ) const { return status!=s.status; }
        VisitedStatus status;
    };

    /// map structure to store a traversal flag for each Node
    typedef std::map<Node*,StatusStruct> StatusMap;

    /// list of Node*
    typedef std::list<Node*> NodeList;

    /// the ordered list of Node to traverse from this Node
    NodeList _precomputedTraversalOrder;

    /// @internal performing only the top-down traversal on a DAG
    /// @executedNodes will be fill with the Nodes where the top-down action is processed
    /// @statusMap the visitor's flag map
    /// @visitorRoot node from where the visitor has been run
    void executeVisitorTopDown(simulation::Visitor* action, NodeList& executedNodes, StatusMap& statusMap, Node* visitorRoot );
    void executeVisitorBottomUp(simulation::Visitor* action, NodeList& executedNodes );
    /// @}

    /// @internal tree traversal implementation
    void executeVisitorTreeTraversal( Visitor* action, StatusMap& statusMap, Visitor::TreeTraversalRepetition repeat, bool alreadyRepeated=false );

    /// @name @internal stuff related to getObjects
    /// @{
    friend class GetDownObjectsVisitor ;
    friend class GetUpObjectsVisitor ;
    /// @}
};

}

