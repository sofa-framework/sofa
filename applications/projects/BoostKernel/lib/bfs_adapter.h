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
#include "BoostSystem.h"
#include "BoostSceneGraph.h"
#include <sofa/component/System.h>
#include <sofa/simulation/tree/Visitor.h>

/**
Adapt a sofa visitor to breadth-first search in a bgl scene graph.

	@author The SOFA team </www.sofa-framework.org>
*/
class bfs_adapter : public boost::bfs_visitor<>
{
public:
    sofa::simulation::tree::Visitor* visitor;

    typedef BoostSceneGraph::BoostGraph Graph; ///< BGL graph to traverse
    BoostSceneGraph::SystemMap& systemMap;      ///< access the System*

    bfs_adapter( sofa::simulation::tree::Visitor* v, BoostSceneGraph::SystemMap& s );

    ~bfs_adapter();

    void discover_vertex( Graph::vertex_descriptor u, const Graph &) const;

    void finish_vertex(Graph::vertex_descriptor u, const Graph &) const;
};

#endif
