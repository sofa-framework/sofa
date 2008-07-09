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
#include "BglScene.h"
#include <sofa/core/objectmodel/ClassInfo.h>


namespace sofa
{
namespace simulation
{
namespace bgl
{

using sofa::core::objectmodel::BaseObject;

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
    BglNode(BglScene* sg, BglScene::Hgraph* g,  BglScene::Hvertex n, const std::string& name="" );
    ~BglNode();

    /** Perform a scene graph traversal with the given Visitor, starting from this node.
    Visitor::processNodetopdown is applied on discover, and Visitor::processNodeBottomUp is applied on finish.
    */
    void doExecuteVisitor( Visitor* action);


    // to move to simulation::Node
    void clearInteractionForceFields();

    /// Generic list of objects access, possibly searching up or down from the current context
    /// @todo Would better be a member of BglScene
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const;



protected:
    BglScene* scene;              ///< the scene the node belongs to
    BglScene::Hgraph* graph;      ///< the mechanical graph of the scene it belongs to
    BglScene::Hvertex vertexId;  ///< its id in the mechanical graph

};

}
}
}

#endif
