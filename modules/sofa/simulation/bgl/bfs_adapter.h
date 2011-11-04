/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
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
#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/simulation/bgl/BglGraphManager.h>
#include <sofa/simulation/common/Visitor.h>
#include <map>

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
class SOFA_SIMULATION_BGL_API  bfs_adapter : public boost::default_bfs_visitor
{
public:
    typedef typename Graph::vertex_descriptor   Vertex;
    typedef typename Graph::edge_descriptor Edge;
    typedef typename boost::property_map<Graph, BglGraphManager::bglnode_t>::type NodeMap;
    typedef typename boost::graph_traits< Graph >::out_edge_iterator OutEdgeIterator;


    bfs_adapter( sofa::simulation::Visitor* v, Graph &g, std::stack< Vertex > &queue):visitor(v), graph(g), visitedNode(queue) {};

    ~bfs_adapter() {};

    /// Applies visitor->processNodeTopDown
    void discover_vertex(Vertex u, const Graph &g)
    {
        Node *node=const_cast<Node*>(get(BglGraphManager::bglnode_t(),g,u));
        visitedNode.push(u);
#ifdef SOFA_DUMP_VISITOR_INFO
        visitor->setNode(node);
        visitor->printInfo(node->getContext(),true);
#endif
        if (visitor->processNodeTopDown(node)==Visitor::RESULT_PRUNE)
        {
            OutEdgeIterator it,it_end;
            for (tie(it, it_end)=out_edges(u, g); it!=it_end;)
            {
                Edge e=*it;
                ++it;
                hack.insert(std::make_pair(u,target(e,g)));
                remove_edge(e,graph);
            }
        }
    }

    void finish_vertex(Vertex v, const Graph &)
    {
        typename std::multimap< Vertex, Vertex >::iterator it, it_end;
        boost::tie(it, it_end)=hack.equal_range(v);
        for (; it!=it_end; ++it)
        {
            add_edge(v,it->second, graph);
        }
        hack.erase(v);
    }

    /// Applies visitor->processNodeBottomUp
    void processBottomUp(Vertex u, const Graph &g)
    {
        Node *node=const_cast<Node*>(get(BglGraphManager::bglnode_t(),g,u));

        visitor->processNodeBottomUp(node);

#ifdef SOFA_DUMP_VISITOR_INFO
        visitor->printInfo(node->getContext(), false);
#endif
    }

    void endTraversal()
    {
        while (!visitedNode.empty())
        {
            Vertex u=visitedNode.top();
            processBottomUp(u, graph);
            visitedNode.pop();
        }
    }

protected:
    sofa::simulation::Visitor* visitor;
    std::multimap< Vertex, Vertex > hack;
    Graph &graph;
    std::stack< Vertex > &visitedNode;
};

}
}
}

#endif
