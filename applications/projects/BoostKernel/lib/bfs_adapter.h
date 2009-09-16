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
// C++ Interface: bfs_adapter
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef bfs_adapter_h
#define bfs_adapter_h

#include <boost/graph/breadth_first_search.hpp>
#include "BglNode.h"
#include "BglGraphManager.h"
#include <sofa/simulation/common/Visitor.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{

/**
Adapt a sofa visitor to breadth-first search in a bgl mapping scene graph.

	@author The SOFA team </www.sofa-framework.org>
*/
template <typename Graph>
class bfs_adapter : public boost::default_bfs_visitor
{
public:
    typedef typename Graph::vertex_descriptor   Vertex;
    typedef typename Graph::edge_descriptor Edge;
    typedef typename boost::property_map<Graph, BglGraphManager::bglnode_t>::type NodeMap;
    typedef typename boost::graph_traits< Graph >::out_edge_iterator OutEdgeIterator;


    bfs_adapter( sofa::simulation::Visitor* v, Graph &g):visitor(v), graph(g)
    {};

    ~bfs_adapter() {};

    /// Applies visitor->processNodeTopDown
    void discover_vertex(Vertex u, const Graph &g) const
    {
        Node *node=const_cast<Node*>(get(BglGraphManager::bglnode_t(),g,u));
        std::cerr << "Visiting " << node->getName() << std::endl;
#ifdef SOFA_DUMP_VISITOR_INFO
        visitor->setNode(node);
        visitor->printInfo(node->getContext(),true);
#endif
        if (visitor->processNodeTopDown(node)==Visitor::RESULT_PRUNE)
        {
            std::cerr << "\tStopping " << node->getName() << std::endl;
            OutEdgeIterator it,it_end;
            for (tie(it, it_end)=out_edges(u, g); it!=it_end;)
            {
                Edge e=*it;
                ++it;
                hack.push_back(std::make_pair(source(e,g), target(e,g)));
                remove_edge(e,graph);
            }
        }
    }

    /// Applies visitor->processNodeBottomUp
    void finish_vertex(Vertex u, const Graph &g) const
    {
        Node *node=const_cast<Node*>(get(BglGraphManager::bglnode_t(),g,u));

        std::cerr << "Ending " << node->getName() << std::endl;
        visitor->processNodeBottomUp(node);

        while (!hack.empty())
        {
            add_edge(hack.front().first,hack.front().second, graph);
            hack.pop_front();
        }

#ifdef SOFA_DUMP_VISITOR_INFO
        visitor->printInfo(node->getContext(), false);
#endif
    }

protected:
    sofa::simulation::Visitor* visitor;
    mutable std::list< std::pair< Vertex, Vertex> > hack;
    mutable Graph &graph;
};

}
}
}

#endif
