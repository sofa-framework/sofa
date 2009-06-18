/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Interface: BglNode
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_BGL_BGLNODE_H
#define SOFA_SIMULATION_BGL_BGLNODE_H

#include "BglGraphManager.h"
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/objectmodel/ClassInfo.h>
#include <sofa/helper/vector.h>


namespace sofa
{
namespace simulation
{
namespace bgl
{

using sofa::core::objectmodel::BaseObject;

using sofa::simulation::Node;
/**
sofa::simulation::Node as a node of a BGL scene graph.


	@author Francois Faure in The SOFA team </www.sofa-framework.org>
*/

class BglNode : public sofa::simulation::Node
{
public:
    typedef sofa::simulation::Visitor Visitor;

    BglNode(BglGraphManager* s,const std::string& name);
    /**
    \param sg the SOFA scene containing a bgl graph
    \param n the node of the bgl graph corresponding to this
    */
    BglNode(BglGraphManager* s, BglGraphManager::Hgraph* g,  BglGraphManager::Hvertex n, const std::string& name="" );
    ~BglNode();

    /** Perform a scene graph traversal with the given Visitor, starting from this node.
    Visitor::processNodetopdown is applied on discover, and Visitor::processNodeBottomUp is applied on finish.
    */
    void doExecuteVisitor( Visitor* action);

    /// Do one step forward in time
    void animate( double dt );


    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const;

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
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const;

    /// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const;

    /// Generic list of objects access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const
    {
        getObjects(class_info, container, sofa::core::objectmodel::TagSet(), dir);
    }

    bool addObject(BaseObject* obj);
    bool removeObject(BaseObject* obj);

    /// Add a child node
    void addChild(core::objectmodel::BaseNode* node);

    /// Remove a child node
    void removeChild(core::objectmodel::BaseNode* node);

    /// Move a node from another node
    void moveChild(core::objectmodel::BaseNode* obj);


    /// Remove the current node from the graph: consists in removing the link to all the parents
    void detachFromGraph() ;


    /// Find all the Nodes pointing
    helper::vector< BglNode* > getParents() const;

    std::string getPathName() const;

    /// Mechanical Degrees-of-Freedom
    virtual core::objectmodel::BaseObject* getMechanicalState() const;

    /// Topology
    virtual core::componentmodel::topology::Topology* getTopology() const;

    /// Mesh Topology (unified interface for both static and dynamic topologies)
    virtual core::componentmodel::topology::BaseMeshTopology* getMeshTopology() const;

    /// Shader
    virtual core::objectmodel::BaseObject* getShader() const;



    /// return the mechanical graph of the scene it belongs to
    BglGraphManager::Hgraph &getGraph() { return *graph;};

    /// return the id of the node in the mechanical graph
    BglGraphManager::Hvertex getVertexId() { return vertexId;};


    /// Called during initialization to corectly propagate the visual context to the children
    virtual void initVisualContext();

    /// Update the whole context values, based on parent and local ContextObjects
    virtual void updateContext();

    /// Update the visual context values, based on parent and local ContextObjects
    virtual void updateVisualContext(int FILTER=0);

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    virtual void updateSimulationContext();



    BglGraphManager* graphManager;              ///< the scene the node belongs to
    BglGraphManager::Hgraph* graph;      ///< the mechanical graph of the scene it belongs to
    BglGraphManager::Hvertex vertexId;  ///< its id in the mechanical graph
    Sequence<BglNode> parents;
    typedef Sequence<BglNode>::iterator ParentIterator;
};

}
}
}

#endif
