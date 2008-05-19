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
#include "BglScene.h"
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
class bfs_adapter : public boost::bfs_visitor<>
{
public:
    sofa::simulation::Visitor* visitor;

    typedef BglScene::Hgraph Graph; ///< BGL graph to traverse
    BglScene::H_vertex_node_map& systemMap;      ///< access the System*

    bfs_adapter( sofa::simulation::tree::Visitor* v, BglScene::H_vertex_node_map& s );

    ~bfs_adapter();

    void discover_vertex( Graph::vertex_descriptor u, const Graph &) const;

    void finish_vertex(Graph::vertex_descriptor u, const Graph &) const;
};

}
}
}

#endif
