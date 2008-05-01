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
#include "BoostSceneGraph.h"
#include "BoostSystem.h"
#include <sofa/simulation/tree/Visitor.h>

/**
Adapt a sofa visitor to depth-first search in a bgl scene graph.

	@author The SOFA team </www.sofa-framework.org>
*/
class dfs_adapter : public boost::dfs_visitor<>
{
public:
    sofa::simulation::tree::Visitor* visitor;

    typedef BoostSceneGraph::BoostGraph Graph; ///< BGL graph to traverse
    BoostSceneGraph::SystemMap& systemMap;      ///< access the System*

    dfs_adapter( sofa::simulation::tree::Visitor* v, BoostSceneGraph::SystemMap& s );

    ~dfs_adapter();

    void discover_vertex( Graph::vertex_descriptor u, const Graph &) const;
    void finish_vertex(Graph::vertex_descriptor u, const Graph &) const;

};

#endif
