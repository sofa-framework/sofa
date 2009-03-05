/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "BglGraphManager.h"
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

The BglGraphManager::H_vertex_node_map is used to get the sofa::simulation::Node associated with a bgl vertex.

	@author The SOFA team </www.sofa-framework.org>
*/
class dfv_adapter : public boost::dfs_visitor<>
{
public:
    sofa::simulation::Visitor* visitor;

    BglGraphManager *graphManager;

    typedef BglGraphManager::Hgraph Graph; ///< BGL graph to traverse
    typedef Graph::vertex_descriptor Vertex;

    BglGraphManager::H_vertex_node_map& systemMap;      ///< access the System*

    dfv_adapter( sofa::simulation::Visitor* v, BglGraphManager *simu,  BglGraphManager::H_vertex_node_map& s );

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
