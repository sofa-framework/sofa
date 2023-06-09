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
#include <sofa/simulation/Node.h>

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/objectmodel/ContextObject.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseTopologyObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/visual/Shader.h>

#include <sofa/core/BehaviorModel.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/Mapping.h>

#include <sofa/simulation/Node.inl>
#include <sofa/simulation/VisitorScheduler.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/DeactivatedNodeVisitor.h>
#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/VisualVisitor.h>

#include <sofa/simulation/MutationListener.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.inl>
#include <sofa/helper/cast.h>
#include <iostream>

/// If you want to activate/deactivate that please set them to true/false
#define DEBUG_VISITOR false
#define DEBUG_LINK false

namespace sofa::simulation
{
using core::objectmodel::BaseNode;
using core::objectmodel::BaseObject;

Node::Node(const std::string& name)
    : core::objectmodel::BaseNode()
    , sofa::core::objectmodel::Context()
    , child(initLink("child", "Child nodes"))
    , object(initLink("object","All objects attached to this node"))

    , behaviorModel(initLink("behaviorModel", "The BehaviorModel attached to this node (only valid for root node)"))
    , mapping(initLink("mapping", "The (non-mechanical) Mapping(s) attached to this node (only valid for root node)"))

    , solver(initLink("odeSolver", "The OdeSolver(s) attached to this node (controlling the mechanical time integration of this branch)"))
    , constraintSolver(initLink("constraintSolver", "The ConstraintSolver(s) attached to this node"))
    , linearSolver(initLink("linearSolver", "The LinearSolver(s) attached to this node"))
    , topologyObject(initLink("topologyObject", "The topology-related objects attached to this node"))
    , forceField(initLink("forceField", "The (non-interaction) ForceField(s) attached to this node"))
    , interactionForceField(initLink("interactionForceField", "The InteractionForceField(s) attached to this node"))
    , projectiveConstraintSet(initLink("projectiveConstraintSet", "The ProjectiveConstraintSet(s) attached to this node"))
    , constraintSet(initLink("constraintSet", "The ConstraintSet(s) attached to this node"))
    , contextObject(initLink("contextObject", "The ContextObject(s) attached to this node"))
    , configurationSetting(initLink("configurationSetting", "The ConfigurationSetting(s) attached to this node"))
    , shaders(initLink("shaders", "The shaders attached to this node"))
    , visualModel(initLink("visualModel", "The VisualModel(s) attached to this node"))
    , visualManager(initLink("visualManager", "The VisualManager(s) attached to this node"))
    , collisionModel(initLink("collisionModel", "The CollisionModel(s) attached to this node"))
    , unsorted(initLink("unsorted", "The remaining objects attached to this node"))

    , animationManager(initLink("animationLoop","The AnimationLoop attached to this node (only valid for root node)"))
    , visualLoop(initLink("visualLoop", "The VisualLoop attached to this node (only valid for root node)"))
    , topology(initLink("topology", "The Topology attached to this node"))
    , meshTopology(initLink("meshTopology", "The MeshTopology / TopologyContainer attached to this node"))
    , state(initLink("state", "The State attached to this node (storing vectors such as position, velocity)"))
    , mechanicalState(initLink("mechanicalState", "The MechanicalState attached to this node (storing all state vectors)"))
    , mechanicalMapping(initLink("mechanicalMapping", "The MechanicalMapping attached to this node"))
    , mass(initLink("mass", "The Mass attached to this node"))
    , collisionPipeline(initLink("collisionPipeline", "The collision Pipeline attached to this node"))

    , debug_(false)
    , initialized(false)
{
    _context = this;
    setName(name);
    f_printLog.setValue(DEBUG_LINK);
}


Node::~Node()
{
}

void Node::parse( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    Inherit1::parse( arg );
    static const char* oldVisualFlags[] =
    {
        "showAll",
        "showVisual",
        "showBehavior",
        "showCollision",
        "showMapping",
        "showVisualModels",
        "showBehaviorModels",
        "showCollisionModels",
        "showBoundingCollisionModels",
        "showMappings",
        "showMechanicalMappings",
        "showForceFields",
        "showInteractionForceFields",
        "showWireFrame",
        "showNormals",
        nullptr
    };
    std::string oldFlags;
    for (unsigned int i=0; oldVisualFlags[i]; ++i)
    {
        const char* str = arg->getAttribute(oldVisualFlags[i], nullptr);
        if (str == nullptr || !*str) continue;
        bool val;
        if (str[0] == 'T' || str[0] == 't')
            val = true;
        else if (str[0] == 'F' || str[0] == 'f')
            val = false;
        else if ((str[0] >= '0' && str[0] <= '9') || str[0] == '-')
            val = (atoi(str) != 0);
        else continue;
        if (!oldFlags.empty()) oldFlags += ' ';
        if (val) oldFlags += oldVisualFlags[i];
        else { oldFlags += "hide"; oldFlags += oldVisualFlags[i]+4; }
    }
    if (!oldFlags.empty())
    {
        msg_deprecated() << "Deprecated visual flags attributes used. Instead, add the following object within the Node: " << msgendl
                         << "<VisualStyle displayFlags=\"" << oldFlags << "\" />" ;

        sofa::core::objectmodel::BaseObjectDescription objDesc("displayFlags","VisualStyle");
        objDesc.setAttribute("displayFlags", oldFlags);
        sofa::core::objectmodel::BaseObject::SPtr obj = sofa::core::ObjectFactory::CreateObject(this, &objDesc);
    }
}

/// Initialize the components of this node and all the nodes which depend on it.
void Node::init(const core::ExecParams* params)
{
    execute<simulation::InitVisitor>(params);
}

/// ReInitialize the components of this node and all the nodes which depend on it.
void Node::reinit(const core::ExecParams* params)
{
    sofa::simulation::DeactivationVisitor deactivate(params, isActive());
    deactivate.execute( this );
}

void Node::draw(core::visual::VisualParams* vparams)
{
    execute<simulation::VisualUpdateVisitor>(vparams);
    execute<simulation::VisualDrawVisitor>(vparams);
}

void Node::addChild(core::objectmodel::BaseNode::SPtr node)
{
    notifyBeginAddChild(this, dynamic_cast<Node*>(node.get()));
    doAddChild(node);
    notifyEndAddChild(this, dynamic_cast<Node*>(node.get()));
}

/// Remove a child
void Node::removeChild(core::objectmodel::BaseNode::SPtr node)
{
    // If node has no parent
    if (node->getFirstParent() == nullptr)
        return;
    notifyBeginRemoveChild(this, static_cast<Node*>(node.get()));
    doRemoveChild(node);
    notifyEndRemoveChild(this, static_cast<Node*>(node.get()));
}


/// Move a node from another node
void Node::moveChild(BaseNode::SPtr node, BaseNode::SPtr prev_parent)
{
    if (!prev_parent.get())
    {
        msg_error(this->getName()) << "Node::moveChild(BaseNode::SPtr node)\n" << node->getName() << " has no parent. Use addChild instead!";
        addChild(node);
        return;
    }
    doMoveChild(node, prev_parent);
}
/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
bool Node::addObject(BaseObject::SPtr obj, sofa::core::objectmodel::TypeOfInsertion insertionLocation)
{
    // If an object we are trying to add already has a context, it is in another node in the
    // graph: we need to remove it from this context before to insert it into the current
    // one.
    if(obj->getContext() != BaseContext::getDefault())
    {
        msg_error() << "Object '" << obj->getName() << "' already has a node ("<< obj->getPathName() << "). Please remove it from this node before adding it to a new one.";
        return false;
    }

    notifyBeginAddObject(this, obj);
    const bool ret = doAddObject(obj, insertionLocation);
    notifyEndAddObject(this, obj);
    return ret;
}

/// Remove an object
bool Node::removeObject(BaseObject::SPtr obj)
{
    notifyBeginRemoveObject(this, obj);
    const bool ret = doRemoveObject(obj);
    notifyEndRemoveObject(this, obj);
    return ret;
}

/// Move an object from another node
void Node::moveObject(BaseObject::SPtr obj)
{
    Node* prev_parent = down_cast<Node>(obj->getContext()->toBaseNode());
    if (prev_parent)
    {
        doMoveObject(obj, prev_parent);
    }
    else
    {
        obj->getContext()->removeObject(obj);
        addObject(obj);
    }
}


void Node::notifyBeginAddChild(Node::SPtr parent, Node::SPtr child) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onBeginAddChild(parent.get(), child.get());
}

void Node::notifyEndAddChild(Node::SPtr parent, Node::SPtr child) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onEndAddChild(parent.get(), child.get());
}

void Node::notifyBeginRemoveChild(Node::SPtr parent, Node::SPtr child) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onBeginRemoveChild(parent.get(), child.get());
}

void Node::notifyEndRemoveChild(Node::SPtr parent, Node::SPtr child) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onEndRemoveChild(parent.get(), child.get());
}

void Node::notifyBeginAddObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr obj) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onBeginAddObject(parent.get(), obj.get());
}

void Node::notifyEndAddObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr obj) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onEndAddObject(parent.get(), obj.get());
}

void Node::notifyBeginRemoveObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr obj) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onBeginRemoveObject(parent.get(), obj.get());
}

void Node::notifyEndRemoveObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr obj) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onEndRemoveObject(parent.get(), obj.get());
}

void Node::notifyBeginAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onBeginAddSlave(master, slave);
}

void Node::notifyEndAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onEndAddSlave(master, slave);
}

void Node::notifyBeginRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onBeginRemoveSlave(master, slave);
}

void Node::notifyEndRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& listener : root->listener)
        listener->onEndRemoveSlave(master, slave);
}

void Node::notifySleepChanged(Node* node) const
{
    if (this->getFirstParent() == nullptr) {
        for (type::vector<MutationListener*>::const_iterator it = listener.begin(); it != listener.end(); ++it)
            (*it)->sleepChanged(node);
    }
    else {
        dynamic_cast<Node*>(this->getFirstParent())->notifySleepChanged(node);
    }
}

void Node::addListener(MutationListener* obj)
{
    // make sure we don't add the same listener twice
    type::vector< MutationListener* >::iterator it = listener.begin();
    while (it != listener.end() && (*it)!=obj)
        ++it;
    if (it == listener.end())
        listener.push_back(obj);
}

void Node::removeListener(MutationListener* obj)
{
    type::vector< MutationListener* >::iterator it = listener.begin();
    while (it != listener.end() && (*it)!=obj)
        ++it;
    if (it != listener.end())
        listener.erase(it);
}


/// Find an object given its name
core::objectmodel::BaseObject* Node::getObject(const std::string& name) const
{
    for (ObjectIterator it = object.begin(), itend = object.end(); it != itend; ++it)
        if ((*it)->getName() == name)
            return it->get();
    return nullptr;
}

sofa::core::objectmodel::Base* Node::findLinkDestClass(const core::objectmodel::BaseClass* destType, const std::string& path, const core::objectmodel::BaseLink* link)
{
    std::string pathStr;
    if (link)
    {
        if (!link->parseString(path,&pathStr))
            return nullptr;
    }
    else
    {
        if (!BaseLink::ParseString(path,&pathStr,nullptr,this))
            return nullptr;
    }

    if(DEBUG_LINK)
        dmsg_info() << "LINK: Looking for " << destType->className << "<" << destType->templateName << "> " << pathStr << " from Node " << getName() ;

    std::size_t ppos = 0;
    const std::size_t psize = pathStr.size();
    if (ppos == psize || (ppos == psize-2 && pathStr[ppos] == '[' && pathStr[ppos+1] == ']')) // self-reference
    {
        if(DEBUG_LINK)
            dmsg_info() << "  self-reference link." ;

        if (!link || !link->getOwnerBase()) return destType->dynamicCast(this);
        return destType->dynamicCast(link->getOwnerBase());
    }
    Node* node = this;
    BaseObject* master = nullptr;
    bool based = false;
    if (ppos < psize && pathStr[ppos] == '[') // relative index in the list of objects
    {
        if (pathStr[psize-1] != ']')
        {
            msg_error() << "Invalid index-based path \"" << path << "\"";
            return nullptr;
        }
        int index = atoi(pathStr.c_str()+ppos+1);

        if(DEBUG_LINK)
           dmsg_info() << "  index-based path to " << index ;

        ObjectReverseIterator it = object.rbegin();
        const ObjectReverseIterator itend = object.rend();
        if (link && link->getOwnerBase())
        {
            // index from last
            Base* b = link->getOwnerBase();
            while (it != itend && *it != b)
                ++it;
        }
        while (it != itend && index < 0)
        {
            ++it;
            ++index;
        }
        if (it == itend)
            return nullptr;

        if(DEBUG_LINK)
            dmsg_info() << "  found " << it->get()->getTypeName() << " " << it->get()->getName() << "." ;

        return destType->dynamicCast(it->get());
    }
    else if (ppos < psize && pathStr[ppos] == '/') // absolute path
    {
        if(DEBUG_LINK)
            dmsg_info() << "  absolute path" ;
        BaseNode* basenode = this->getRoot();
        if (!basenode) return nullptr;
        node = down_cast<Node>(basenode);
        ++ppos;
        based = true;
    }
    while(ppos < psize)
    {
        if ((ppos+1 < psize && pathStr.substr(ppos,2) == "./")
            || pathStr.substr(ppos) == ".")
        {
            // this must be this node
            if(DEBUG_LINK)
                dmsg_info() << "  to current node" ;

            ppos += 2;
            based = true;
        }
        else if ((ppos+2 < psize && pathStr.substr(ppos,3) == "../") // relative
                || pathStr.substr(ppos) == "..")
        {
            ppos += 3;
            if (master)
            {
                master = master->getMaster();
                if(DEBUG_LINK)
                    dmsg_info() << "  to master object " << master->getName() ;
            }
            else
            {
                core::objectmodel::BaseNode* firstParent = node->getFirstParent();
                if (!firstParent) return nullptr;
                node = static_cast<Node*>(firstParent); // TODO: explore other parents
                if(DEBUG_LINK)
                    dmsg_info() << "  to parent node " << node->getName() ;
            }
            based = true;
        }
        else if (pathStr[ppos] == '/')
        {
            // extra /
            if(DEBUG_LINK)
                dmsg_info() << "  extra '/'" ;
            ppos += 1;
        }
        else
        {
            std::size_t p2pos = pathStr.find('/',ppos);
            if (p2pos == std::string::npos) p2pos = psize;
            std::string name = pathStr.substr(ppos,p2pos-ppos);
            ppos = p2pos+1;
            if (master)
            {
                if(DEBUG_LINK)
                    dmsg_info() << "  to slave object " << name ;
                master = master->getSlave(name);
                if (!master) return nullptr;
            }
            else
            {
                for (;;)
                {
                    BaseObject* obj = node->getObject(name);
                    Node* child = node->getChild(name);
                    if (child)
                    {
                        node = child;
                        if(DEBUG_LINK)
                            dmsg_info() << "  to child node " << name ;
                        break;
                    }
                    else if (obj)
                    {
                        master = obj;
                        if(DEBUG_LINK)
                            dmsg_info()  << "  to object " << name ;
                        break;
                    }
                    if (based) return nullptr;
                    // this can still be found from an ancestor node
                    core::objectmodel::BaseNode* firstParent = node->getFirstParent();
                    if (!firstParent) return nullptr;
                    node = static_cast<Node*>(firstParent); // TODO: explore other parents
                    if(DEBUG_LINK)
                        dmsg_info()  << "  looking in ancestor node " << node->getName() ;
                }
            }
            based = true;
        }
    }
    if (master)
    {
        if(DEBUG_LINK)
            dmsg_info()  << "  found " << master->getTypeName() << " " << master->getName() << "." ;
        return destType->dynamicCast(master);
    }
    else
    {
        Base* r = destType->dynamicCast(node);
        if (r)
        {
            if(DEBUG_LINK)
                dmsg_info()  << "  found node " << node->getName() << "." ;
            return r;
        }
        for (ObjectIterator it = node->object.begin(), itend = node->object.end(); it != itend; ++it)
        {
            BaseObject* obj = it->get();
            Base *o = destType->dynamicCast(obj);
            if (!o) continue;
            if(DEBUG_LINK)
                dmsg_info()  << "  found " << obj->getTypeName() << " " << obj->getName() << "." ;
            if (!r) r = o;
            else return nullptr; // several objects are possible, this is an ambiguous path
        }
        if (r) return r;
        // no object found, we look in parent nodes if the searched class is one of the known standard single components (state, topology, ...)
        if (destType->hasParent(sofa::core::BaseState::GetClass()))
            return destType->dynamicCast(node->getState());
        else if (destType->hasParent(core::topology::BaseMeshTopology::GetClass()))
            return destType->dynamicCast(node->getMeshTopologyLink());
        else if (destType->hasParent(core::topology::Topology::GetClass()))
            return destType->dynamicCast(node->getTopology());
        else if (destType->hasParent(core::visual::Shader::GetClass()))
            return destType->dynamicCast(node->getShader());
        else if (destType->hasParent(core::behavior::BaseAnimationLoop::GetClass()))
            return destType->dynamicCast(node->getAnimationLoop());
        else if (destType->hasParent(core::behavior::OdeSolver::GetClass()))
            return destType->dynamicCast(node->getOdeSolver());
        else if (destType->hasParent(core::collision::Pipeline::GetClass()))
            return destType->dynamicCast(node->getCollisionPipeline());
        else if (destType->hasParent(core::visual::VisualLoop::GetClass()))
            return destType->dynamicCast(node->getVisualLoop());

        return nullptr;
    }
}

/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
bool Node::doAddObject(BaseObject::SPtr sobj, sofa::core::objectmodel::TypeOfInsertion insertionLocation)
{
    this->setObjectContext(sobj);
    if(insertionLocation == sofa::core::objectmodel::TypeOfInsertion::AtEnd)
        object.add(sobj);
    else
        object.addBegin(sobj);

    BaseObject* obj = sobj.get();

    if( !obj->insertInNode( this ) )
    {
        unsorted.add(obj);
    }
    return true;
}

/// Remove an object
bool Node::doRemoveObject(BaseObject::SPtr sobj)
{
    dmsg_warning_when(sobj == nullptr) << "Trying to remove a nullptr object";

    this->clearObjectContext(sobj);
    object.remove(sobj);
    BaseObject* obj = sobj.get();

    if(obj != nullptr && !obj->removeInNode( this ) )
        unsorted.remove(obj);
    return true;
}

/// Remove an object
void Node::doMoveObject(BaseObject::SPtr sobj, Node* prev_parent)
{
    if (prev_parent != nullptr)
        prev_parent->removeObject(sobj);
    addObject(sobj);
}


/// Topology
core::topology::Topology* Node::getTopology() const
{
    if (this->topology)
        return this->topology;
    else
        return get<core::topology::Topology>(SearchParents);
}

/// Mesh Topology (unified interface for both static and dynamic topologies)
core::topology::BaseMeshTopology* Node::getMeshTopologyLink(SearchDirection dir) const
{
    SOFA_UNUSED(dir);
    if (this->meshTopology)
        return this->meshTopology;
    else
        return get<core::topology::BaseMeshTopology>(SearchParents);
}

/// Degrees-of-Freedom
core::BaseState* Node::getState() const
{
    if (this->state)
        return this->state;
    else
        return get<core::BaseState>(SearchParents);
}

/// Mechanical Degrees-of-Freedom
core::behavior::BaseMechanicalState* Node::getMechanicalState() const
{
    if (this->mechanicalState)
        return this->mechanicalState;
    else
        return get<core::behavior::BaseMechanicalState>(SearchParents);
}

/// Shader
core::visual::Shader* Node::getShader() const
{
    if (!shaders.empty())
        return *shaders.begin();
    else
        return get<core::visual::Shader>(SearchParents);
}
core::visual::Shader* Node::getShader(const sofa::core::objectmodel::TagSet& t) const
{
    if(t.empty())
        return getShader();
    else // if getShader is Tag filtered
    {
        for(NodeSequence<core::visual::Shader>::iterator it = shaders.begin(), iend=shaders.end(); it!=iend; ++it)
        {
            if ( (*it)->getTags().includes(t) )
                return (*it);
        }
        return get<core::visual::Shader>(t,SearchParents);
    }
}

core::behavior::BaseAnimationLoop* Node::getAnimationLoop() const
{
    if (animationManager)
        return animationManager;
    else
        return get<core::behavior::BaseAnimationLoop>(SearchParents);
}

core::behavior::OdeSolver* Node::getOdeSolver() const
{
    if (!solver.empty())
        return solver[0];
    else
        return get<core::behavior::OdeSolver>(SearchParents);
}

core::collision::Pipeline* Node::getCollisionPipeline() const
{
    if (collisionPipeline)
        return collisionPipeline;
    else
        return get<core::collision::Pipeline>(SearchParents);
}

core::visual::VisualLoop* Node::getVisualLoop() const
{
    if (visualLoop)
        return visualLoop;
    else
        return get<core::visual::VisualLoop>(SearchParents);
}

/// Find a child node given its name
Node* Node::getChild(const std::string& name) const
{
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
    {
        if ((*it)->getName() == name)
            return it->get();
    }
    return nullptr;
}

/// Get a descendant node given its name
Node* Node::getTreeNode(const std::string& name) const
{
    Node* result = nullptr;
    result = getChild(name);
    for (ChildIterator it = child.begin(), itend = child.end(); result == nullptr && it != itend; ++it)
        result = (*it)->getTreeNode(name);
    return result;
}

Node* Node::getNodeInGraph(const std::string& absolutePath) const
{
    if (absolutePath[0] != '/')
        return nullptr;

    std::string p = absolutePath.substr(1);
    if (p == "")
        return dynamic_cast<Node*>(this->getRootContext());

    Node* ret = nullptr;
    const Node* parent = dynamic_cast<Node*>(this->getRootContext());
    while (p != "")
    {
        std::string nodeName = p.substr(0, p.find('/'));
        ret = parent->getChild(nodeName);
        if (!ret)
            return nullptr;
        if (p.find('/') == std::string::npos)
            p = "";
        else
            p = p.substr(p.find('/') +1);
        parent = ret;
    }
    if (!ret)
        return nullptr;
    return ret;
}

/// Get parent node (or nullptr if no hierarchy or for root node)
sofa::core::objectmodel::BaseNode::Children Node::getChildren() const
{
    Children list_children;
    list_children.reserve(child.size());
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
        list_children.push_back(it->get());
    return list_children;
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
    typedef NodeSequence<core::behavior::OdeSolver> Solvers;
    const Solvers solverRemove = solver;
    for ( Solvers::iterator i=solverRemove.begin(), iend=solverRemove.end(); i!=iend; ++i )
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


void Node::setDefaultVisualContextValue()
{
    //TODO(dmarchal 2017-07-20) please say who have to do that and when it will be done.
    /// @todo: This method is now broken because getShow*() methods never return -1
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
    */
}

void Node::initialize()
{
    initialized = true;  // flag telling is the node is initialized

    initVisualContext();
    updateSimulationContext();
}

void Node::updateContext()
{
    updateSimulationContext();
    updateVisualContext();
    if ( debug_ )
        msg_info()<<"Node::updateContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) ;
}

void Node::updateSimulationContext()
{
    for ( unsigned i=0; i<contextObject.size(); ++i )
    {
        contextObject[i]->init();
        contextObject[i]->apply();
    }
}

void Node::updateVisualContext()
{
    // Apply local modifications to the context
    for ( unsigned i=0; i<contextObject.size(); ++i )
    {
        contextObject[i]->init();
        contextObject[i]->apply();
    }

    if ( debug_ )
        msg_info()<<"Node::updateVisualContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) ;
}

/// Execute a recursive action starting from this node
void Node::executeVisitor(Visitor* action, bool precomputedOrder)
{
    if (!this->isActive()) return;
    // if the current node is sleeping and the visitor can't access it, don't do anything
    if (this->isSleeping() && !action->canAccessSleepingNode) return;

    static int level = 0;
    if(DEBUG_VISITOR)
    {
        std::stringstream tmp;
        for (int i=0; i<level; ++i)
            tmp << ' ';
        tmp << ">" << sofa::helper::NameDecoder::decodeClassName(typeid(*action)) << " on " << this->getPathName();
        if (!action->getInfos().empty())
            tmp << "  : " << action->getInfos();
        dmsg_info () << tmp.str() ;
        ++level;
    }

    doExecuteVisitor(action, precomputedOrder);

    if(DEBUG_VISITOR)
    {
        --level;
        std::stringstream tmp;
        for (int i=0; i<level; ++i)
            tmp << ' ';
        tmp  << "<" << sofa::helper::NameDecoder::decodeClassName(typeid(*action)) << " on " << this->getPathName();
        dmsg_info() << tmp.str() ;
    }
}

/// Propagate an event
void Node::propagateEvent(const core::ExecParams* params, core::objectmodel::Event* event)
{
    simulation::PropagateEventVisitor act(params, event);
    this->executeVisitor(&act);
}





void Node::printComponents()
{
    using namespace sofa::core::behavior;
    using core::BaseMapping;
    using core::topology::Topology;
    using core::topology::BaseTopologyObject;
    using core::topology::BaseMeshTopology;
    using core::visual::Shader;
    using core::BehaviorModel;
    using core::visual::VisualModel;
    using core::visual::VisualLoop;
    using core::CollisionModel;
    using core::objectmodel::ContextObject;
    using core::collision::Pipeline;
    using core::BaseState;

    std::stringstream sstream;

    sstream << "BaseAnimationLoop: ";
    for (NodeSingle<BaseAnimationLoop>::iterator i = animationManager.begin(), iend = animationManager.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "OdeSolver: ";
    for (NodeSequence<OdeSolver>::iterator i = solver.begin(), iend = solver.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "LinearSolver: ";
    for (NodeSequence<BaseLinearSolver>::iterator i = linearSolver.begin(), iend = linearSolver.end(); i != iend; i++)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "ConstraintSolver: ";
    for (NodeSequence<ConstraintSolver>::iterator i = constraintSolver.begin(), iend = constraintSolver.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "VisualLoop: ";
    for (NodeSingle<VisualLoop>::iterator i = visualLoop.begin(), iend = visualLoop.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "InteractionForceField: ";
    for (NodeSequence<BaseInteractionForceField>::iterator i = interactionForceField.begin(), iend = interactionForceField.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "ForceField: ";
    for (NodeSequence<BaseForceField>::iterator i = forceField.begin(), iend = forceField.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "State: ";
    for (NodeSingle<BaseState>::iterator i = state.begin(), iend = state.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "MechanicalState: ";
    for (NodeSingle<BaseMechanicalState>::iterator i = mechanicalState.begin(), iend = mechanicalState.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "Mechanical Mapping: ";
    for (NodeSingle<BaseMapping>::iterator i = mechanicalMapping.begin(), iend = mechanicalMapping.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "Mapping: ";
    for (NodeSequence<BaseMapping>::iterator i = mapping.begin(), iend = mapping.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "Topology: ";
    for (NodeSingle<Topology>::iterator i = topology.begin(), iend = topology.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "MeshTopology: ";
    for (NodeSingle<BaseMeshTopology>::iterator i = meshTopology.begin(), iend = meshTopology.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "Shader: ";
    for (NodeSequence<Shader>::iterator i = shaders.begin(), iend = shaders.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "ProjectiveConstraintSet: ";
    for (NodeSequence<BaseProjectiveConstraintSet>::iterator i = projectiveConstraintSet.begin(), iend = projectiveConstraintSet.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "ConstraintSet: ";
    for (NodeSequence<BaseConstraintSet>::iterator i = constraintSet.begin(), iend = constraintSet.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "BehaviorModel: ";
    for (NodeSequence<BehaviorModel>::iterator i = behaviorModel.begin(), iend = behaviorModel.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "VisualModel: ";
    for (NodeSequence<VisualModel>::iterator i = visualModel.begin(), iend = visualModel.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "CollisionModel: ";
    for (NodeSequence<CollisionModel>::iterator i = collisionModel.begin(), iend = collisionModel.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "ContextObject: ";
    for (NodeSequence<ContextObject>::iterator i = contextObject.begin(), iend = contextObject.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n" << "Pipeline: ";
    for (NodeSingle<Pipeline>::iterator i = collisionPipeline.begin(), iend = collisionPipeline.end(); i != iend; ++i)
        sstream << (*i)->getName() << " ";
    sstream << "\n";

    msg_info() << sstream.str();
}

Node::SPtr Node::create( const std::string& name )
{
    if (Simulation* simulation = getSimulation())
    {
        return simulation->createNewNode(name);
    }
    return nullptr;
}

void Node::setSleeping(bool val)
{
    if (val != d_isSleeping.getValue())
    {
        d_isSleeping.setValue(val);
        notifySleepChanged(this);
    }
}

#define NODE_DEFINE_SEQUENCE_ACCESSOR( CLASSNAME, FUNCTIONNAME, SEQUENCENAME ) \
    void Node::add##FUNCTIONNAME( CLASSNAME* obj ) { SEQUENCENAME.add(obj); } \
    void Node::remove##FUNCTIONNAME( CLASSNAME* obj ) { SEQUENCENAME.remove(obj); }

NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseAnimationLoop, AnimationLoop, animationManager )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::visual::VisualLoop, VisualLoop, visualLoop )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::BehaviorModel, BehaviorModel, behaviorModel )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::BaseMapping, Mapping, mapping )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::OdeSolver, OdeSolver, solver )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::ConstraintSolver, ConstraintSolver, constraintSolver )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseLinearSolver, LinearSolver, linearSolver )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::topology::Topology, Topology, topology )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::topology::BaseMeshTopology, MeshTopology, meshTopology )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::topology::BaseTopologyObject, TopologyObject, topologyObject )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::BaseState, State, state )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseMechanicalState,MechanicalState, mechanicalState )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::BaseMapping, MechanicalMapping, mechanicalMapping )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseMass, Mass, mass )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseForceField, ForceField, forceField )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseInteractionForceField, InteractionForceField, interactionForceField )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseProjectiveConstraintSet, ProjectiveConstraintSet, projectiveConstraintSet )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::behavior::BaseConstraintSet, ConstraintSet, constraintSet )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::objectmodel::ContextObject, ContextObject, contextObject )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::objectmodel::ConfigurationSetting, ConfigurationSetting, configurationSetting )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::visual::Shader, Shader, shaders )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::visual::VisualModel, VisualModel, visualModel )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::visual::VisualManager, VisualManager, visualManager )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::CollisionModel, CollisionModel, collisionModel )
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::collision::Pipeline, CollisionPipeline, collisionPipeline )

template class NodeSequence<Node,true>;
template class NodeSequence<sofa::core::objectmodel::BaseObject,true>;
template class NodeSequence<sofa::core::BehaviorModel>;
template class NodeSequence<sofa::core::BaseMapping>;
template class NodeSequence<sofa::core::behavior::OdeSolver>;
template class NodeSequence<sofa::core::behavior::ConstraintSolver>;
template class NodeSequence<sofa::core::behavior::BaseLinearSolver>;
template class NodeSequence<sofa::core::topology::BaseTopologyObject>;
template class NodeSequence<sofa::core::behavior::BaseForceField>;
template class NodeSequence<sofa::core::behavior::BaseInteractionForceField>;
template class NodeSequence<sofa::core::behavior::BaseProjectiveConstraintSet>;
template class NodeSequence<sofa::core::behavior::BaseConstraintSet>;
template class NodeSequence<sofa::core::objectmodel::ContextObject>;
template class NodeSequence<sofa::core::objectmodel::ConfigurationSetting>;
template class NodeSequence<sofa::core::visual::Shader>;
template class NodeSequence<sofa::core::visual::VisualModel>;
template class NodeSequence<sofa::core::visual::VisualManager>;
template class NodeSequence<sofa::core::CollisionModel>;
template class NodeSequence<sofa::core::objectmodel::BaseObject>;

template class NodeSingle<sofa::core::behavior::BaseAnimationLoop>;
template class NodeSingle<sofa::core::visual::VisualLoop>;
template class NodeSingle<sofa::core::topology::Topology>;
template class NodeSingle<sofa::core::topology::BaseMeshTopology>;
template class NodeSingle<sofa::core::BaseState>;
template class NodeSingle<sofa::core::behavior::BaseMechanicalState>;
template class NodeSingle<sofa::core::BaseMapping>;
template class NodeSingle<sofa::core::behavior::BaseMass>;
template class NodeSingle<sofa::core::collision::Pipeline>;

}
