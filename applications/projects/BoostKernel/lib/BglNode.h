/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef BglNode_h
#define BglNode_h

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include "BglSimulation.h"
#include <sofa/core/objectmodel/ClassInfo.h>


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

    /**
    \param sg the SOFA scene containing a bgl graph
    \param n the node of the bgl graph corresponding to this
    */
    BglNode(BglSimulation* sg, BglSimulation::Hgraph* g,  BglSimulation::Hvertex n, const std::string& name="" );
    ~BglNode();

    /** Perform a scene graph traversal with the given Visitor, starting from this node.
    Visitor::processNodetopdown is applied on discover, and Visitor::processNodeBottomUp is applied on finish.
    */
    void doExecuteVisitor( Visitor* action);


    // to move to simulation::Node
    void clearInteractionForceFields();


    /// Generic object access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, SearchDirection dir = SearchUp) const;

    /// Generic object access, given a path from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const;


    /// Generic list of objects access, possibly searching up or down from the current context
    /// @todo Would better be a member of BglSimulation
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const;


    /// Add a child node
    void addChild(Node* node);

    /// Remove a child node
    void removeChild(Node* node);

    /// Move a node from another node
    void moveChild(Node* obj);

    /// return the mechanical graph of the scene it belongs to
    BglSimulation::Hgraph &getGraph() { return *graph;};

    /// return the id of the node in the mechanical graph
    BglSimulation::Hvertex getVertexId() { return vertexId;};

protected:
    BglSimulation* scene;              ///< the scene the node belongs to
    BglSimulation::Hgraph* graph;      ///< the mechanical graph of the scene it belongs to
    BglSimulation::Hvertex vertexId;  ///< its id in the mechanical graph

};

}
}
}

#endif
