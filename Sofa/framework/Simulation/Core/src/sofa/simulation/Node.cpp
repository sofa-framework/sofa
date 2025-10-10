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
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
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

Node::Node(const std::string& nodename, Node* parent)
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
    , visualStyle(initLink("visualStyle", "The VisualStyle(s) attached to this node"))
    , topology(initLink("topology", "The Topology attached to this node"))
    , meshTopology(initLink("meshTopology", "The MeshTopology / TopologyContainer attached to this node"))
    , state(initLink("state", "The State attached to this node (storing vectors such as position, velocity)"))
    , mechanicalState(initLink("mechanicalState", "The MechanicalState attached to this node (storing all state vectors)"))
    , mechanicalMapping(initLink("mechanicalMapping", "The MechanicalMapping attached to this node"))
    , mass(initLink("mass", "The Mass attached to this node"))
    , collisionPipeline(initLink("collisionPipeline", "The collision Pipeline attached to this node"))

    , debug_(false)
    , initialized(false)
    , l_parents(initLink("parents", "Parents nodes in the graph"))
{
    if( parent )
        parent->addChild(this);

    _context = this;
    setName(nodename);
    f_printLog.setValue(DEBUG_LINK);
}


Node::~Node()
{
    for (auto& aChild : child )
        aChild->l_parents.remove(this);
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


void Node::notifyBeginAddChild(Node::SPtr parent, Node::SPtr childPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onBeginAddChild(parent.get(), childPtr.get());
}

void Node::notifyEndAddChild(Node::SPtr parent, Node::SPtr childPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onEndAddChild(parent.get(), childPtr.get());
}

void Node::notifyBeginRemoveChild(Node::SPtr parent, Node::SPtr childPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onBeginRemoveChild(parent.get(), childPtr.get());
}

void Node::notifyEndRemoveChild(Node::SPtr parent, Node::SPtr childPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onEndRemoveChild(parent.get(), childPtr.get());
}

void Node::notifyBeginAddObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr objPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onBeginAddObject(parent.get(), objPtr.get());
}

void Node::notifyEndAddObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr objPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onEndAddObject(parent.get(), objPtr.get());
}

void Node::notifyBeginRemoveObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr objPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onBeginRemoveObject(parent.get(), objPtr.get());
}

void Node::notifyEndRemoveObject(Node::SPtr parent, core::objectmodel::BaseObject::SPtr objPtr) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onEndRemoveObject(parent.get(), objPtr.get());
}

void Node::notifyBeginAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onBeginAddSlave(master, slave);
}

void Node::notifyEndAddSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onEndAddSlave(master, slave);
}

void Node::notifyBeginRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onBeginRemoveSlave(master, slave);
}

void Node::notifyEndRemoveSlave(core::objectmodel::BaseObject* master, core::objectmodel::BaseObject* slave) const
{
    const Node* root = down_cast<Node>(this->getContext()->getRootContext()->toBaseNode());
    for (const auto& rootListener : root->listener)
        rootListener->onEndRemoveSlave(master, slave);
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
core::objectmodel::BaseObject* Node::getObject(const std::string& objectName) const
{
    for (ObjectIterator it = object.begin(), itend = object.end(); it != itend; ++it)
        if ((*it)->getName() == objectName)
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
            std::string nameStr = pathStr.substr(ppos,p2pos-ppos);
            ppos = p2pos+1;
            if (master)
            {
                if(DEBUG_LINK)
                    dmsg_info() << "  to slave object " << nameStr ;
                master = master->getSlave(nameStr);
                if (!master) return nullptr;
            }
            else
            {
                for (;;)
                {
                    BaseObject* obj = node->getObject(nameStr);
                    Node* childPtr = node->getChild(nameStr);
                    if (childPtr)
                    {
                        node = childPtr;
                        if(DEBUG_LINK)
                            dmsg_info() << "  to child node " << nameStr ;
                        break;
                    }
                    else if (obj)
                    {
                        master = obj;
                        if(DEBUG_LINK)
                            dmsg_info()  << "  to object " << nameStr ;
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
Node* Node::getChild(const std::string& childName) const
{
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
    {
        if ((*it)->getName() == childName)
            return it->get();
    }
    return nullptr;
}

/// Get a descendant node given its name
Node* Node::getTreeNode(const std::string& nodeName) const
{
    Node* result = nullptr;
    result = getChild(nodeName);
    for (ChildIterator it = child.begin(), itend = child.end(); result == nullptr && it != itend; ++it)
        result = (*it)->getTreeNode(nodeName);
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

void Node::updateVisualContext()
{
    initializeContexts();

    dmsg_info_when(debug_)<<"Node::updateVisualContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) ;
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
    using core::visual::BaseVisualStyle;

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
    sstream << "\n" << "BaseVisualStyle: ";
    for (NodeSingle<BaseVisualStyle>::iterator i = visualStyle.begin(), iend = visualStyle.end(); i != iend; ++i)
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

void Node::setSleeping(bool val)
{
    if (val != d_isSleeping.getValue())
    {
        d_isSleeping.setValue(val);
        notifySleepChanged(this);
    }
}

template<class LinkType, class Component>
void checkAlreadyContains(Node& self, LinkType& link, Component* obj)
{
    if constexpr (!LinkType::IsMultiLink)
    {
        if (link != obj && link != nullptr)
        {
            static const auto componentClassName = Component::GetClass()->className;
            msg_warning(&self) << "Trying to add a " << componentClassName << " ('"
                << obj->getName() << "' [" << obj->getClassName() << "] " << obj << ")"
                << " into the Node '" << self.getPathName()
                << "', whereas it already contains one ('" << link->getName() << "' [" << link->getClassName() << "] " << link.get() << ")."
                << " Only one " << componentClassName << " is permitted in a Node. The previous "
                << componentClassName << " is replaced and the behavior is undefined.";
        }
    }
}


/// get all down objects respecting specified class_info and tags
class GetDownObjectsVisitor : public Visitor
{
public:

    GetDownObjectsVisitor(const sofa::core::objectmodel::ClassInfo& class_info, Node::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags);
    ~GetDownObjectsVisitor() ;

    Result processNodeTopDown(simulation::Node* node) override
    {
        static_cast<const Node*>(node)->getLocalObjects( _class_info, _container, _tags );
        return RESULT_CONTINUE;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return false; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "GetDownObjectsVisitor"; }
    const char* getClassName()    const override { return "GetDownObjectsVisitor"; }

protected:
    const sofa::core::objectmodel::ClassInfo& _class_info;
    Node::GetObjectsCallBack& _container;
    const sofa::core::objectmodel::TagSet& _tags;
};

GetDownObjectsVisitor::GetDownObjectsVisitor(const sofa::core::objectmodel::ClassInfo& class_info,
                                             Node::GetObjectsCallBack& container,
                                             const sofa::core::objectmodel::TagSet& tags)
    : Visitor( sofa::core::execparams::defaultInstance() )
    , _class_info(class_info)
    , _container(container)
    , _tags(tags)
{}

GetDownObjectsVisitor::~GetDownObjectsVisitor(){}

/// get all up objects respecting specified class_info and tags
class GetUpObjectsVisitor : public Visitor
{
public:

    GetUpObjectsVisitor(Node* searchNode, const sofa::core::objectmodel::ClassInfo& class_info, Node::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags);
    ~GetUpObjectsVisitor() override;

    Result processNodeTopDown(simulation::Node* node) override
    {
        const Node* dagnode = dynamic_cast<const Node*>(node);
        if( dagnode->_descendancy.contains(_searchNode) ) // searchNode is in the current node descendancy, so the current node is a parent of searchNode
        {
            dagnode->getLocalObjects( _class_info, _container, _tags );
            return RESULT_CONTINUE;
        }
        else // the current node is NOT a parent of searchNode, stop here
        {
            return RESULT_PRUNE;
        }
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return false; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "GetUpObjectsVisitor"; }
    const char* getClassName()    const override { return "GetUpObjectsVisitor"; }


protected:

    Node* _searchNode;
    const sofa::core::objectmodel::ClassInfo& _class_info;
    Node::GetObjectsCallBack& _container;
    const sofa::core::objectmodel::TagSet& _tags;

};

GetUpObjectsVisitor::GetUpObjectsVisitor(Node* searchNode,
                                         const sofa::core::objectmodel::ClassInfo& class_info,
                                         Node::GetObjectsCallBack& container,
                                         const sofa::core::objectmodel::TagSet& tags)
    : Visitor( sofa::core::execparams::defaultInstance() )
    , _searchNode( searchNode )
    , _class_info(class_info)
    , _container(container)
    , _tags(tags)
{}

GetUpObjectsVisitor::~GetUpObjectsVisitor(){}

/// Create, add, then return the new child of this Node
Node::SPtr Node::createChild(const std::string& nodeName)
{
    Node::SPtr newchild;
    if (nodeName.empty())
    {
        int i = 0;
        std::string newName = "unnamed";
        bool uid_found = false;
        while (!uid_found)
        {
            uid_found = true;
            for (const auto& c : this->child)
            {
                if (c->getName() == newName)
                {
                    newName = "unnamed" + std::to_string(++i);
                    uid_found = true;
                }
            }
            for (const auto& o : this->object)
            {
                if (o->getName() == newName)
                {
                    newName = "unnamed" + std::to_string(++i);
                    uid_found = true;
                }
            }
        }
        msg_error("Node::createChild()") << "Empty string given to property 'name': Forcefully setting an empty name is forbidden.\n"
                                        "Renaming to " + newName + " to avoid unexpected behaviors.";
        newchild = sofa::core::objectmodel::New<Node>(newName);
    }
    else
        newchild = sofa::core::objectmodel::New<Node>(nodeName);
    this->addChild(newchild); newchild->updateSimulationContext();
    return newchild;
}


void Node::moveChild(BaseNode::SPtr node)
{
    const Node::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<Node>(node);
    for (const auto& parent : dagnode->getParents()) {
        Node::moveChild(node, parent);
    }
}


/// Add a child node
void Node::doAddChild(BaseNode::SPtr node)
{
    const Node::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<Node>(node);
    setDirtyDescendancy();
    child.add(dagnode);
    dagnode->l_parents.add(this);
    dagnode->l_parents.updateLinks(); // to fix load-time unresolved links
}

/// Remove a child
void Node::doRemoveChild(BaseNode::SPtr node)
{
    const Node::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<Node>(node);
    setDirtyDescendancy();
    child.remove(dagnode);
    dagnode->l_parents.remove(this);
}

/// Move a node from another node
void Node::doMoveChild(BaseNode::SPtr node, BaseNode::SPtr previous_parent)
{
    const Node::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<Node>(node);
    if (!dagnode) return;

    setDirtyDescendancy();
    previous_parent->removeChild(node);

    addChild(node);
}

/// Remove a child
void Node::detachFromGraph()
{
    Node::SPtr me = this; // make sure we don't delete ourself before the end of this method
    const LinkParents::Container& parents = l_parents.getValue();
    while(!parents.empty())
        parents.back()->removeChild(this);
}

/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* Node::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    if (dir == SearchRoot)
    {
        if (getNbParents()) return getRootContext()->getObject(class_info, tags, dir);
        else dir = SearchDown; // we are the root, search down from here.
    }
    void *result = nullptr;

    if (dir != SearchParents)
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            sofa::core::objectmodel::BaseObject* obj = it->get();
            if (tags.empty() || (obj)->getTags().includes(tags))
            {

                result = class_info.dynamicCast(obj);
                if (result != nullptr)
                {

                    break;
                }
            }
        }

    if (result == nullptr)
    {
        switch(dir)
        {
            case Local:
                break;
            case SearchParents:
            case SearchUp:
            {
                const LinkParents::Container& parents = l_parents.getValue();
                for ( unsigned int i = 0; i < parents.size() ; ++i){
                    result = parents[i]->getObject(class_info, tags, SearchUp);
                    if (result != nullptr) break;
                }
            }
                break;
            case SearchDown:
                for(ChildIterator it = child.begin(); it != child.end(); ++it)
                {
                    result = (*it)->getObject(class_info, tags, dir);
                    if (result != nullptr) break;
                }
                break;
            case SearchRoot:
                dmsg_error("Node") << "SearchRoot SHOULD NOT BE POSSIBLE HERE.";
                break;
        }
    }

    return result;
}

/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* Node::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const
{
    if (path.empty())
    {
        // local object
        return Node::getObject(class_info, Local);
    }
    else if (path[0] == '/')
    {
        // absolute path; let's start from root
        if (!getNbParents()) return getObject(class_info,std::string(path,1));
        else return getRootContext()->getObject(class_info,path);
    }
    else if (std::string(path,0,2)==std::string("./"))
    {
        std::string newpath = std::string(path, 2);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        return getObject(class_info,newpath);
    }
    else if (std::string(path,0,3)==std::string("../"))
    {
        // tricky case:
        // let's test EACH parent and return the first object found (if any)
        std::string newpath = std::string(path, 3);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        if (getNbParents())
        {
            const LinkParents::Container& parents = l_parents.getValue();
            for ( unsigned int i = 0; i < parents.size() ; ++i)
            {
                void* obj = parents[i]->getObject(class_info,newpath);
                if (obj) return obj;
            }
            return nullptr;   // not found in any parent node at all
        }
        else return getObject(class_info,newpath);
    }
    else
    {
        std::string::size_type pend = path.find('/');
        if (pend == std::string::npos) pend = path.length();
        const std::string childName ( path, 0, pend );
        const Node* childPtr = getChild(childName);
        if (childPtr)
        {
            while (pend < path.length() && path[pend] == '/')
                ++pend;
            return childPtr->getObject(class_info, std::string(path, pend));
        }
        else if (pend < path.length())
        {
            return nullptr;
        }
        else
        {
            sofa::core::objectmodel::BaseObject* obj = simulation::Node::getObject(childName);
            if (obj == nullptr)
            {
                return nullptr;
            }
            else
            {
                void* result = class_info.dynamicCast(obj);
                if (result == nullptr)
                {
                    dmsg_error("Node") << "Object "<<childName<<" in "<<getPathName()<<" does not implement class "<<class_info.name() ;
                    return nullptr;
                }
                else
                {
                    return result;
                }
            }
        }
    }
}


/// Generic list of objects access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void Node::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    if( dir == SearchRoot )
    {
        if( getNbParents() )
        {
            getRootContext()->getObjects( class_info, container, tags, dir );
            return;
        }
        else dir = SearchDown; // we are the root, search down from here.
    }


    switch( dir )
    {
        case Local:
            this->getLocalObjects( class_info, container, tags );
            break;

        case SearchUp:
            this->getLocalObjects( class_info, container, tags ); // add locals then SearchParents
            // no break here, we want to execute the SearchParents code.
            [[fallthrough]];
        case SearchParents:
        {
            // a visitor executed from top but only run for this' parents will enforce the selected object unicity due even with diamond graph setups
            GetUpObjectsVisitor vis( const_cast<Node*>(this), class_info, container, tags);
            getRootContext()->executeVisitor(&vis);
        }
            break;

        case SearchDown:
        {
            // a regular visitor is enforcing the selected object unicity
            GetDownObjectsVisitor vis(class_info, container, tags);
            (const_cast<Node*>(this))->executeVisitor(&vis);
            break;
        }
        default:
            break;
    }
}

/// Get a list of parent node
sofa::core::objectmodel::BaseNode::Parents Node::getParents() const
{
    Parents p;

    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
        p.push_back(parents[i]);

    return p;
}


/// returns number of parents
size_t Node::getNbParents() const
{
    return l_parents.getValue().size();
}

/// return the first parent (returns nullptr if no parent)
sofa::core::objectmodel::BaseNode* Node::getFirstParent() const
{
    const LinkParents::Container& parents = l_parents.getValue();
    if( parents.empty() ) return nullptr;
    else return l_parents.getValue()[0];
}


/// Test if the given node is a parent of this node.
bool Node::hasParent(const BaseNode* node) const
{
    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
    {
        if (parents[i]==node) return true;
    }
    return false;
}

/// Test if the given context is a parent of this context.
bool Node::hasParent(const BaseContext* context) const
{
    if (context == nullptr) return !getNbParents();

    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
        if (context == parents[i]->getContext()) return true;
    return false;

}



/// Test if the given context is an ancestor of this context.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool Node::hasAncestor(const BaseContext* context) const
{
    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
        if (context == parents[i]->getContext()
                || parents[i]->hasAncestor(context))
            return true;
    return false;
}


/// Mesh Topology that is relevant for this context
/// (within it or its parents until a mapping is reached that does not preserve topologies).
sofa::core::topology::BaseMeshTopology* Node::getMeshTopologyLink(SearchDirection dir) const
{
    // If there is a topology in the current node
    if (this->meshTopology)
        return this->meshTopology;

    // If we are not forcing on local resolution, search in the parents
    if (dir != Local)
        return get<core::topology::BaseMeshTopology>(SearchParents);

    // At that step there is no local topology or we are doing a non local search (so searching in the parents).

    // TODO(dmarchal, 2025-07-16): Why a mapping interfere with the search for a topology ?
    //                             This is the kind of "implicit" hard coded behavior that generates troubles
    //                             Investigate if this could be removed without too much breaks

    // Check if there is a local mapping and this mapping does not have the same topology so it step the search
    if ( mechanicalMapping && ! mechanicalMapping->sameTopology())
        return nullptr;

    // TODO(dmarchal, 2025-07-16): This tests seems to do exactly the same as the one on MechanicalMapping.
    //                             The test before can probably be removed
    // Check if any of the other mapping does not have the same
    for ( auto& aMapping : mapping)
    {
        if (!aMapping->sameTopology())
            return nullptr;
    }

    // TODO(dmarchal, 2025-07-16): The following code is probably ill-defined, what it does it probably going to search
    // in parents... for a topology, priorizing the "first" parent that returns one. It is kind of strange to search in parent
    // while because at that step the search is: dir == Local ... so searching in parent is just "weird".

    // No mapping with a different topology, continue on to the parents
    const LinkParents::Container &parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++ )
    {
        // if the visitor is run from a sub-graph containing a multinode linked with a node outside of the subgraph, do not consider the outside node by looking on the sub-graph descendancy
        if ( parents[i] )
        {
            sofa::core::topology::BaseMeshTopology* res = parents[i]->getMeshTopologyLink(Local);
            if (res)
                return res;
        }
    }
    return nullptr; // not found in any parents
}

void Node::precomputeTraversalOrder( const sofa::core::ExecParams* params )
{
    // accumulating traversed Nodes
    class TraversalOrderVisitor : public Visitor
    {
        NodeList& _orderList;
    public:
        TraversalOrderVisitor(const sofa::core::ExecParams* eparams, NodeList& orderList )
            : Visitor(eparams)
            , _orderList( orderList )
        {
            _orderList.clear();
        }

        Result processNodeTopDown(Node* node) override
        {
            _orderList.push_back(node);
            return RESULT_CONTINUE;
        }

        const char* getClassName() const override {return "TraversalOrderVisitor";}
    };

    TraversalOrderVisitor tov( params, _precomputedTraversalOrder );
    executeVisitor( &tov, false );
}



/// Execute a recursive action starting from this node
void Node::doExecuteVisitor(simulation::Visitor* action, bool precomputedOrder)
{
    if( precomputedOrder && !_precomputedTraversalOrder.empty() )
    {
        for( NodeList::iterator it = _precomputedTraversalOrder.begin(), itend = _precomputedTraversalOrder.end() ; it != itend ; ++it )
        {
            if ( action->canAccessSleepingNode || !(*it)->getContext()->isSleeping() )
                action->processNodeTopDown( *it );
        }

        for( NodeList::reverse_iterator it = _precomputedTraversalOrder.rbegin(), itend = _precomputedTraversalOrder.rend() ; it != itend ; ++it )
        {
            if ( action->canAccessSleepingNode || !(*it)->getContext()->isSleeping() )
                action->processNodeBottomUp( *it );
        }
    }
    else
    {
        // WARNING: do not store the traversal infos in the Node, as several visitors could traversed the graph simultaneously
        // These infos are stored in a StatusMap per visitor.
        updateDescendancy();

        Visitor::TreeTraversalRepetition repeat;
        if( action->treeTraversal(repeat) )
        {
            // Tree traversal order
            //
            // Diamond shapes are ignored, a child node is visited as soon as a parent node has been visited.
            // The multi-nodes (with several parents) are visited either: only once, only twice or for every times
            // depending on the visitor's 'repeat'
            //
            // Some particular visitors such as a flat graph display or VisualVisitors must follow such a traversal order.

            StatusMap statusMap;
            executeVisitorTreeTraversal( action, statusMap, repeat );
        }
        else
        {
            // Direct acyclic graph traversal order
            //
            // This is the default order, used for mechanics.
            //
            // A child node is visited only when all its parents have been visited.
            // A child node is 'pruned' only if all its parents are 'pruned'.
            // Every executed node in the forward traversal are stored in 'executedNodes',
            // its reverse order is used for the backward traversal.

            // Note that a newly 'pruned' node is still traversed (w/o execution) to be sure to execute its child nodes,
            // that can have ancestors in another branch that is not pruned...
            // An already pruned node is ignored.

            NodeList executedNodes;
            {
                StatusMap statusMap;
                executeVisitorTopDown( action, executedNodes, statusMap, this );
            }
            executeVisitorBottomUp( action, executedNodes );
        }
    }
}


void Node::executeVisitorTopDown(simulation::Visitor* action, NodeList& executedNodes, StatusMap& statusMap, Node* visitorRoot )
{
    if ( statusMap[this] != NOT_VISITED )
    {
        return; // skipped (already visited)
    }

    if( !this->isActive() )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;

        // in that case we can considerer if some child are activated, the graph is not valid, so no need to continue the recursion
        return;
    }

    if( this->isSleeping() && !action->canAccessSleepingNode )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;

        return;
    }

    // pour chaque noeud "prune" on continue à parcourir quand même juste pour marquer le noeud comme parcouru

    // check du "visitedStatus" des parents:
    // un enfant n'est pruné que si tous ses parents le sont
    // on ne passe à un enfant que si tous ses parents ont été visités
    bool allParentsPruned = true;
    bool hasParent = false;

    if( visitorRoot != this )
    {
        // the graph structure is generally modified during an action anterior to the traversal but can possibly be modified during the current traversal
        visitorRoot->updateDescendancy();

        const LinkParents::Container &parents = l_parents.getValue();
        for ( unsigned int i = 0; i < parents.size() ; i++ )
        {
            // if the visitor is run from a sub-graph containing a multinode linked with a node outside of the subgraph, do not consider the outside node by looking on the sub-graph descendancy
            if ( visitorRoot->_descendancy.contains(parents[i]) || parents[i]==visitorRoot )
            {
                // all parents must have been visited before
                if ( statusMap[parents[i]] == NOT_VISITED )
                    return; // skipped for now... the other parent should come later

                allParentsPruned = allParentsPruned && ( statusMap[parents[i]] == PRUNED );
                hasParent = true;
            }
        }
    }

    // all parents have been visited, let's go with the visitor
    if ( allParentsPruned && hasParent )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;

        // ... but continue the recursion anyway!
        if( action->childOrderReversed(this) )
            for(unsigned int i = unsigned(child.size()); i>0;)
                child[--i].get()->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);
        else
            for(unsigned int i = 0; i<child.size(); ++i)
                child[i].get()->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);
    }
    else
    {
        // execute the visitor on this node
        const Visitor::Result result = action->processNodeTopDown(this);

        // update status
        statusMap[this] = ( result == simulation::Visitor::RESULT_PRUNE ? PRUNED : VISITED );

        executedNodes.push_back(this);

        // ... and continue the recursion
        if( action->childOrderReversed(this) )
            for(unsigned int i = unsigned(child.size()); i>0;)
                child[--i].get()->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);
        else
            for(unsigned int i = 0; i<child.size(); ++i)
                child[i].get()->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);

    }
}


// warning nodes that are dynamically created during the traversal, but that have not been traversed during the top-down, won't be traversed during the bottom-up
// TODO is it what we want?
// otherwise it is possible to restart from top, go to leaves and running bottom-up action while going up
void Node::executeVisitorBottomUp( simulation::Visitor* action, NodeList& executedNodes )
{
    for( NodeList::reverse_iterator it = executedNodes.rbegin(), itend = executedNodes.rend() ; it != itend ; ++it )
    {
        (*it)->updateDescendancy();
        action->processNodeBottomUp( *it );
    }
}


void Node::setDirtyDescendancy()
{
    _descendancy.clear();
    const LinkParents::Container &parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++ )
    {
        parents[i]->setDirtyDescendancy();
    }
}

void Node::updateDescendancy()
{
    if( _descendancy.empty() && !child.empty() )
    {
        for(unsigned int i = 0; i<child.size(); ++i)
        {
            Node* node = child[i].get();
            node->updateDescendancy();
            _descendancy.insert( node->_descendancy.begin(), node->_descendancy.end() );
            _descendancy.insert( node );
        }
    }
}



void Node::executeVisitorTreeTraversal( simulation::Visitor* action, StatusMap& statusMap, Visitor::TreeTraversalRepetition repeat, bool alreadyRepeated )
{
    if( !this->isActive() )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;
        return;
    }

    if( this->isSleeping() && !action->canAccessSleepingNode )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;
        return;
    }

    // node already visited and repetition must be avoid
    if( statusMap[this] != NOT_VISITED )
    {
        if( repeat==Visitor::NO_REPETITION || ( alreadyRepeated && repeat==Visitor::REPEAT_ONCE ) ) return;
        else alreadyRepeated = true;
    }

    if( action->processNodeTopDown(this) != simulation::Visitor::RESULT_PRUNE )
    {
        statusMap[this] = VISITED;
        if( action->childOrderReversed(this) )
            for(unsigned int i = unsigned(child.size()); i>0;)
                child[--i].get()->executeVisitorTreeTraversal(action,statusMap,repeat,alreadyRepeated);
        else
            for(unsigned int i = 0; i<child.size(); ++i)
                child[i].get()->executeVisitorTreeTraversal(action,statusMap,repeat,alreadyRepeated);
    }
    else
    {
        statusMap[this] = PRUNED;
    }

    action->processNodeBottomUp(this);
}

void Node::initVisualContext()
{
    if (getNbParents())
    {
        this->setDisplayWorldGravity(false); //only display gravity for the root: it will be propagated at each time step
    }
}

void Node::initializeContexts()
{
    for ( unsigned i=0; i<contextObject.size(); ++i )
    {
        contextObject[i]->init();  // Call init of a ContextObject (a component in the scene)
        contextObject[i]->apply(); // The component copy its internal state in the node's (that is inheriting from Contexte)
    }
}

void Node::updateContext()
{
    // if there is a parent
    sofa::core::objectmodel::BaseNode* firstParent = getFirstParent();
    if ( firstParent )
    {
        dmsg_info_when(debug_)<<"Node::updateContext, node = "<<getName()<<", incoming context = "<< firstParent->getContext() ;

        // TODO (dmarchal, 16-07-2025): There is underlying assumption here that the firstParent is
        //                              the one we want copy the context from. This is an implict behavior
        //                              if one day we refactor that part, maybe it would be better to have
        //                              an an explicit context-relationship and trigger a warning in case like the following one
        //                              saying there is an ambiguity and query scene designer to deambiguiate it.
        copyContext(*static_cast<Context*>(static_cast<Node*>(firstParent)));
    }

    updateSimulationContext();
    updateVisualContext();

    dmsg_info_when(debug_)<<"Node::updateContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) ;
}

void Node::updateSimulationContext()
{
    // if there is a parent
    sofa::core::objectmodel::BaseNode* firstParent = getFirstParent();
    if ( firstParent )
    {
        dmsg_info_when(debug_)<<"Node::updateSimulationContext, node = "<<getName()<<", incoming context = "<< firstParent->getContext() ;

        // TODO (dmarchal, 16-07-2025): There is underlying assumption here that the firstParent is
        //                              the one we want copy the context from. This is an implict behavior
        //                              if one day we refactor that part, maybe it would be better to have
        //                              an an explicit context-relationship and trigger a warning in case like the following one
        //                              saying there is an ambiguity and query scene designer to deambiguiate it.
        copySimulationContext(*static_cast<Context*>(static_cast<Node*>(firstParent)));
    }

    // if there is no parent... initialize all the context objects.
    // TODO (dmarchal, 16-07-2025): It is weird to actually initialize something at update. A carfull investigation is probably worth
    initializeContexts();
}

Node* Node::findCommonParent( simulation::Node* node2 )
{
    return static_cast<Node*>(getRoot())->findCommonParent(this, node2);
}

Node* Node::findCommonParent(Node* node1, Node* node2)
{
    updateDescendancy();

    if (!_descendancy.contains(node1) || !_descendancy.contains(node2))
        return nullptr; // this is NOT a parent

    // this is a parent
    for (unsigned int i = 0; i<child.size(); ++i)
    {
        // look for closer parents
        Node* childcommon = child[i].get()->findCommonParent(node1, node2);

        if (childcommon != nullptr)
            return childcommon;
    }
    // NO closer parents found
    return this;
}

void Node::getLocalObjects( const sofa::core::objectmodel::ClassInfo& class_info, Node::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags ) const
{
    for (Node::ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
    {
        sofa::core::objectmodel::BaseObject* obj = it->get();
        void* result = class_info.dynamicCast(obj);
        if (result != nullptr && (tags.empty() || (obj)->getTags().includes(tags)))
            container(result);
    }
}

#define NODE_DEFINE_SEQUENCE_ACCESSOR( CLASSNAME, FUNCTIONNAME, SEQUENCENAME ) \
    void Node::add##FUNCTIONNAME( CLASSNAME* obj ) { checkAlreadyContains(*this, SEQUENCENAME, obj); SEQUENCENAME.add(obj); } \
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
NODE_DEFINE_SEQUENCE_ACCESSOR( sofa::core::visual::BaseVisualStyle, VisualStyle, visualStyle )
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
template class NodeSingle<sofa::core::visual::BaseVisualStyle>;
template class NodeSingle<sofa::core::topology::Topology>;
template class NodeSingle<sofa::core::topology::BaseMeshTopology>;
template class NodeSingle<sofa::core::BaseState>;
template class NodeSingle<sofa::core::behavior::BaseMechanicalState>;
template class NodeSingle<sofa::core::BaseMapping>;
template class NodeSingle<sofa::core::behavior::BaseMass>;
template class NodeSingle<sofa::core::collision::Pipeline>;

}
