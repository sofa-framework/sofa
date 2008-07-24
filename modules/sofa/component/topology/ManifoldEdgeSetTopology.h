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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGY_H

#include <sofa/component/topology/EdgeSetTopology.h>

#include <sofa/component/topology/ManifoldEdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/ManifoldEdgeSetTopologyAlgorithms.h>
#include <sofa/component/topology/ManifoldEdgeSetTopologyModifier.h>
#include <sofa/component/topology/ManifoldEdgeSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class ManifoldEdgeSetTopology;

class ManifoldEdgeSetTopologyContainer;

template<class DataTypes>
class ManifoldEdgeSetTopologyModifier;

template < class DataTypes >
class ManifoldEdgeSetTopologyAlgorithms;

template < class DataTypes >
class ManifoldEdgeSetGeometryAlgorithms;

template <class DataTypes>
class ManifoldEdgeSetTopologyLoader;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;
typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::VertexEdges VertexEdges;

/** Describes a topological object that only consists as a set of points and lines connecting these points.
    This topology is constraint by the manifold property : each vertex is adjacent either to one vertex or to two vertices. */
template<class DataTypes>
class ManifoldEdgeSetTopology : public EdgeSetTopology <DataTypes>
{
public:

    ManifoldEdgeSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~ManifoldEdgeSetTopology() {}

    virtual void init();

    /** \brief Returns the EdgeSetTopologyContainer object of this ManifoldEdgeSetTopology.
     */
    ManifoldEdgeSetTopologyContainer *getEdgeSetTopologyContainer() const
    {
        return static_cast<ManifoldEdgeSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the EdgeSetTopologyModifier object of this ManifoldEdgeSetTopology.
    */
    ManifoldEdgeSetTopologyModifier<DataTypes> *getEdgeSetTopologyModifier() const
    {
        return static_cast<ManifoldEdgeSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the EdgeSetTopologyAlgorithms object of this ManifoldEdgeSetTopology.
     */
    ManifoldEdgeSetTopologyAlgorithms<DataTypes> *getEdgeSetTopologyAlgorithms() const
    {
        return static_cast<ManifoldEdgeSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Generic method returning the TopologyAlgorithms object // BIBI
    */
    /*
    virtual core::componentmodel::topology::TopologyAlgorithms *getTopologyAlgorithms() const {
    return getEdgeSetTopologyAlgorithms();
    }
    */

    /** \brief Returns the EdgeSetGeometryAlgorithms object of this ManifoldEdgeSetTopology.
    */
    ManifoldEdgeSetGeometryAlgorithms<DataTypes> *getEdgeSetGeometryAlgorithms() const
    {
        return static_cast<ManifoldEdgeSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /// BaseMeshTopology API
    /// @{

    virtual const SeqEdges& getEdges()
    {
        return getEdgeSetTopologyContainer()->getEdgeArray();
    }

    /// Returns the set of edges adjacent to a given vertex.
    virtual const VertexEdges& getEdgeVertexShell(PointID i)
    {
        return getEdgeSetTopologyContainer()->getEdgeVertexShell(i);
    }

    /// Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
    virtual int getEdgeIndex(PointID v1, PointID v2)
    {
        return getEdgeSetTopologyContainer()->getEdgeIndex(v1, v2);
    }

    /// @}

protected:
    virtual void createComponents();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
