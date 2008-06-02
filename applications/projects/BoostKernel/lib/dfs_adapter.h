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
#include "BglScene.h"
#include "BglNode.h"
#include <sofa/simulation/common/Visitor.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{

/**
Adapt a sofa visitor to a depth-first search in a bgl graph encoding the mechanical mapping hierarchy.
The BglScene::H_vertex_node_map is used to get the sofa::simulation::Node associated with a bgl vertex.

	@author The SOFA team </www.sofa-framework.org>
*/
class dfs_adapter : public boost::dfs_visitor<>
{
public:
    sofa::simulation::Visitor* visitor;

    typedef BglScene::Hgraph Graph; ///< BGL graph to traverse
    typedef Graph::vertex_descriptor Vertex;

    BglScene::H_vertex_node_map& systemMap;      ///< access the System*

    dfs_adapter( sofa::simulation::Visitor* v, BglScene::H_vertex_node_map& s );

    ~dfs_adapter();

    /// Applies visitor->processNodeTopDown
    void discover_vertex (Vertex u, const Graph &);

    /// Applies visitor->processNodeBottomUp
    void finish_vertex(Vertex u, const Graph &) const;

};
}
}
}


#endif
