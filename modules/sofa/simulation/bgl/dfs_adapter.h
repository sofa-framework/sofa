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
// C++ Interface: dfs_adapter
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef dfs_adapter_h
#define dfs_adapter_h

#include <boost/graph/depth_first_search.hpp>
#include <sofa/simulation/bgl/BglGraphManager.h>
#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/simulation/common/Visitor.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{

/**
Adapt a sofa visitor to a depth-first search in a bgl graph encoding the mechanical mapping hierarchy.

	@author The SOFA team </www.sofa-framework.org>
*/
template <typename Graph>
class SOFA_SIMULATION_BGL_API  dfs_adapter : public boost::dfs_visitor<>
{
public:
    typedef typename Graph::vertex_descriptor Vertex;

    dfs_adapter( sofa::simulation::Visitor* v ):visitor(v) {};

    ~dfs_adapter() {};

    /// Applies visitor->processNodeTopDown
    void discover_vertex(Vertex u, const Graph &g)
    {
        Node *node=const_cast<Node*>(get(BglGraphManager::bglnode_t(),g,u));
#ifdef SOFA_DUMP_VISITOR_INFO
        visitor->setNode(node);
        visitor->printInfo(node->getContext(),true);
#endif
        visitor->processNodeTopDown(node);
    }

    /// Applies visitor->processNodeBottomUp
    void finish_vertex(Vertex u, const Graph &g) const
    {
        Node *node=const_cast<Node*>(get(BglGraphManager::bglnode_t(),g,u));
        visitor->processNodeBottomUp(node);
#ifdef SOFA_DUMP_VISITOR_INFO
        visitor->printInfo(node->getContext(), false);
#endif
    }

protected:
    sofa::simulation::Visitor* visitor;

};
}
}
}


#endif
