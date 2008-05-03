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
#include "BglSystem.h"
#include <sofa/simulation/tree/Visitor.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{

/**
Adapt a sofa visitor to depth-first search in a bgl mapping graph.

	@author The SOFA team </www.sofa-framework.org>
*/
class dfs_adapter : public boost::dfs_visitor<>
{
public:
    sofa::simulation::tree::Visitor* visitor;

    typedef BglScene::MappingGraph Graph; ///< BGL graph to traverse
    BglScene::SystemMap& systemMap;      ///< access the System*

    dfs_adapter( sofa::simulation::tree::Visitor* v, BglScene::SystemMap& s );

    ~dfs_adapter();

    void discover_vertex( Graph::vertex_descriptor u, const Graph &) const;
    void finish_vertex(Graph::vertex_descriptor u, const Graph &) const;

};
}
}
}


#endif
