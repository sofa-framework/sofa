//
// C++ Interface: dfv_adapter
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef dfv_adapter_h
#define dfv_adapter_h

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
Adapt a sofa visitor to a depth-first visit in a bgl graph encoding the mechanical mapping hierarchy.
This visitor is aimed to be used by a depth first visit with a pruning criterion.
The criterion is embedded within the dfv_adapter.
The usual method discover_vertex is replaced by operator(), used by the bgl to evaluate if the visit must be pruned.
Operator () calls visitor->processNodeTopDown and returns true iff this method has returned RESULT_PRUNE.

The BglScene::H_vertex_node_map is used to get the sofa::simulation::Node associated with a bgl vertex.

	@author The SOFA team </www.sofa-framework.org>
*/
class dfv_adapter : public boost::dfs_visitor<>
{
public:
    sofa::simulation::Visitor* visitor;

    typedef BglScene::Hgraph Graph; ///< BGL graph to traverse
    typedef Graph::vertex_descriptor Vertex;

    BglScene::H_vertex_node_map& systemMap;      ///< access the System*

    dfv_adapter( sofa::simulation::Visitor* v, BglScene::H_vertex_node_map& s );

    ~dfv_adapter();

    /// Applies visitor->processNodeTopDown
    bool operator() (Vertex u, const Graph &);

    /// Applies visitor->processNodeBottomUp
    void finish_vertex(Vertex u, const Graph &) const;

};
}
}
}


#endif
