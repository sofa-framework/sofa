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
// C++ Implementation: BglGraphManager
//
// Description:
//
//
// Author: Francois Faure in The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_SIMULATION_BGL_BGLGRAPHMANAGER_INL
#define SOFA_SIMULATION_BGL_BGLGRAPHMANAGER_INL

#include <sofa/simulation/bgl/BglGraphManager.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{

template <typename Container>
void BglGraphManager::getParentNodes(Container &data, const Node* node)
{
    getInVertices(data, h_node_vertex_map[const_cast<Node*>(node)], hgraph);
}

template <typename Container>
void BglGraphManager::getChildNodes(Container &data, const Node* node)
{
    getOutVertices(data, h_node_vertex_map[const_cast<Node*>(node)], hgraph);
}


template <typename Container,  typename Graph>
void BglGraphManager::getInVertices(Container &data, typename Graph::vertex_descriptor v, Graph &g)
{
    typedef typename boost::graph_traits< Graph >::in_edge_iterator InEdgeIterator;
    typedef typename boost::graph_traits< Graph >::vertex_descriptor Vertex;

    InEdgeIterator it,it_end;
    for (tie(it, it_end)=in_edges(v, g); it!=it_end; ++it)
    {
        Node *inV=getNode(source(*it, g), g);
        data.insert(data.end(),static_cast<typename Container::value_type>(inV));
    }
}




template <typename Container, typename Graph>
void BglGraphManager::getOutVertices(Container &data, typename Graph::vertex_descriptor v, Graph &g)
{
    typedef typename boost::graph_traits< Graph >::out_edge_iterator OutEdgeIterator;
    typedef typename boost::graph_traits< Graph >::vertex_descriptor Vertex;

    OutEdgeIterator it,it_end;
    for (tie(it, it_end)=out_edges(v, g); it!=it_end; ++it)
    {
        Node *outV=getNode(target(*it, g), g);
        data.insert(data.end(),static_cast<typename Container::value_type>(outV));
    }
}

template <typename Container>
void BglGraphManager::getRoots(Container &data)
{
    HvertexVector::iterator it, it_end=hroots.end();
    for (it=hroots.begin(); it!=it_end; ++it) data.push_back(getNode(*it, hgraph));
}

template <typename Graph>
Node *BglGraphManager::getNode( typename Graph::vertex_descriptor v, Graph &g)
{
    return get( bglnode_t(), g, v);
}




}
}
}

#endif
