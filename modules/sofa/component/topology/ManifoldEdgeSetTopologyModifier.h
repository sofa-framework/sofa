/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYMODIFIER_H

#include <sofa/component/topology/EdgeSetTopologyModifier.h>

namespace sofa
{

namespace component
{

namespace topology
{
class ManifoldEdgeSetTopologyContainer;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;
typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::VertexEdges VertexEdges;

/**
* A class that can apply basic transformations on a set of points.
*/
class ManifoldEdgeSetTopologyModifier : public EdgeSetTopologyModifier
{
public:
    ManifoldEdgeSetTopologyModifier()
        : EdgeSetTopologyModifier()
    {}

    virtual ~ManifoldEdgeSetTopologyModifier() {}

    virtual void init();

    /** \brief Add an edge.
    */
    void addEdgeProcess(Edge e);

    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Effectively Remove a subset of edges. Eventually remove isolated vertices
    *
    * Elements corresponding to these edges are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the edges are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    */
    virtual void removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedItems = false);

private:
    ManifoldEdgeSetTopologyContainer* 	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
