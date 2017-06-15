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
#ifndef SOFA_CORE_OBJECTMODEL_BASENODE_H
#define SOFA_CORE_OBJECTMODEL_BASENODE_H

#include "BaseContext.h"
#include "BaseObject.h"

namespace sofa
{

namespace core
{

// forward declaration of classes accessible from the node
namespace behavior
{
class BaseAnimationLoop;
class OdeSolver;
}
namespace collision
{
class Pipeline;
}
namespace visual
{
class VisualLoop;
}

namespace objectmodel
{

/**
 *  \brief Base class for simulation nodes.
 *
 *  A Node is a class defining the main scene data structure of a simulation.
 *  It defined hierarchical relations between elements.
 *  Each node can have parent and child nodes (potentially defining a tree),
 *  as well as attached objects (the leaves of the tree).
 *
 * \author Jeremie Allard
 */
class SOFA_CORE_API BaseNode : public virtual Base
{
public:
    SOFA_ABSTRACT_CLASS(BaseNode, Base);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseNode)

protected:
    BaseNode() ;
    virtual ~BaseNode();

private:
    BaseNode(const BaseNode& n) ;
    BaseNode& operator=(const BaseNode& n) ;
public:
    /// @name Scene hierarchy
    /// @{

    typedef sofa::helper::vector< BaseNode* > Children;
    /// Get a list of child node
    virtual Children getChildren() const = 0;

    typedef sofa::helper::vector< BaseNode* > Parents;
    /// Get a list of parent node
    /// @warning a temporary is created, this can be really inefficient
    virtual Parents getParents() const = 0;

    /// returns number of parents
    virtual size_t getNbParents() const = 0;

    /// return the first parent (returns NULL if no parent)
    virtual BaseNode* getFirstParent() const = 0;

    /// returns the root by following up the first parent for multinodes
    virtual BaseNode* getRoot() const;

    /// Add a child node
    virtual void addChild(BaseNode::SPtr node) = 0;

    /// Remove a child node
    virtual void removeChild(BaseNode::SPtr node) = 0;

    /// Move a node from another node
    virtual void moveChild(BaseNode::SPtr node) = 0;

    /// Add a generic object
    virtual bool addObject(BaseObject::SPtr obj) = 0;

    /// Remove a generic object
    virtual bool removeObject(BaseObject::SPtr obj) = 0;

    /// Move an object from a node to another node
    virtual void moveObject(BaseObject::SPtr obj) = 0;

    /// Test if the given node is a parent of this node.
    virtual bool hasParent(const BaseNode* node) const = 0;

    /// Test if the given node is an ancestor of this node.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    virtual bool hasAncestor(const BaseNode* node) const = 0;

    /// Remove the current node from the graph: depending on the type of Node, it can have one or several parents.
    /// \warning You must have a SPtr on the node you detach if you want to keep it or the smart pointer mechanism will remove it !
    virtual void detachFromGraph() = 0;

    /// Get this node context
    virtual BaseContext* getContext() = 0;

    /// Get this node context
    virtual const BaseContext* getContext() const = 0;

    /// Return the full path name of this node
    virtual std::string getPathName() const;

    /// Return the path from this node to the root node
    virtual std::string getRootPath() const;

    virtual void* findLinkDestClass(const BaseClass* destType, const std::string& path, const BaseLink* link) = 0;

    /// @}

    /// @name Solvers and main algorithms
    /// @{

    virtual core::behavior::BaseAnimationLoop* getAnimationLoop() const;
    virtual core::behavior::OdeSolver* getOdeSolver() const;
    virtual core::collision::Pipeline* getCollisionPipeline() const;
    virtual core::visual::VisualLoop* getVisualLoop() const;

    /// @}
protected:
    /// Set the context of an object to this
    void setObjectContext(BaseObject::SPtr obj);

    /// Reset the context of an object
    void clearObjectContext(BaseObject::SPtr obj);


    /// @name virtual functions to add/remove special components direclty in the right Sequence
    /// Note it is useful for Node, but is not mandatory for every BaseNode Inheritances
    /// so the default implementation does nothing
    /// @{

#define BASENODE_ADD_SPECIAL_COMPONENT( CLASSNAME, FUNCTIONNAME, SEQUENCENAME ) \
    virtual void add##FUNCTIONNAME( CLASSNAME* ) {} \
    virtual void remove##FUNCTIONNAME( CLASSNAME* ) {}

public:

     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseAnimationLoop, AnimationLoop, animationManager )
     BASENODE_ADD_SPECIAL_COMPONENT( core::visual::VisualLoop, VisualLoop, visualLoop )
     BASENODE_ADD_SPECIAL_COMPONENT( core::BehaviorModel, BehaviorModel, behaviorModel )
     BASENODE_ADD_SPECIAL_COMPONENT( core::BaseMapping, Mapping, mapping )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::OdeSolver, OdeSolver, solver )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::ConstraintSolver, ConstraintSolver, constraintSolver )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseLinearSolver, LinearSolver, linearSolver )
     BASENODE_ADD_SPECIAL_COMPONENT( core::topology::Topology, Topology, topology )
     BASENODE_ADD_SPECIAL_COMPONENT( core::topology::BaseMeshTopology, MeshTopology, meshTopology )
     BASENODE_ADD_SPECIAL_COMPONENT( core::topology::BaseTopologyObject, TopologyObject, topologyObject )
     BASENODE_ADD_SPECIAL_COMPONENT( core::BaseState, State, state )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseMechanicalState,MechanicalState, mechanicalState )
     BASENODE_ADD_SPECIAL_COMPONENT( core::BaseMapping, MechanicalMapping, mechanicalMapping )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseMass, Mass, mass )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseForceField, ForceField, forceField )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseInteractionForceField, InteractionForceField, interactionForceField )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseProjectiveConstraintSet, ProjectiveConstraintSet, projectiveConstraintSet )
     BASENODE_ADD_SPECIAL_COMPONENT( core::behavior::BaseConstraintSet, ConstraintSet, constraintSet )
     BASENODE_ADD_SPECIAL_COMPONENT( core::objectmodel::ContextObject, ContextObject, contextObject )
     BASENODE_ADD_SPECIAL_COMPONENT( core::objectmodel::ConfigurationSetting, ConfigurationSetting, configurationSetting )
     BASENODE_ADD_SPECIAL_COMPONENT( core::visual::Shader, Shader, shaders )
     BASENODE_ADD_SPECIAL_COMPONENT( core::visual::VisualModel, VisualModel, visualModel )
     BASENODE_ADD_SPECIAL_COMPONENT( core::visual::VisualManager, VisualManager, visualManager )
     BASENODE_ADD_SPECIAL_COMPONENT( core::CollisionModel, CollisionModel, collisionModel )
     BASENODE_ADD_SPECIAL_COMPONENT( core::collision::Pipeline, CollisionPipeline, collisionPipeline )

#undef BASENODE_ADD_SPECIAL_COMPONENT

    /// @}

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
