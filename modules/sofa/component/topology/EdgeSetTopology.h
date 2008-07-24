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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_H

#include <sofa/component/topology/PointSetTopology.h>

#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/EdgeSetTopologyAlgorithms.h>
#include <sofa/component/topology/EdgeSetTopologyModifier.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/EdgeSetTopologyChange.h>

namespace sofa
{

namespace component
{

namespace topology
{
template<class DataTypes>
class EdgeSetTopology;

class EdgeSetTopologyContainer;

template<class DataTypes>
class EdgeSetTopologyModifier;

template < class DataTypes >
class EdgeSetTopologyAlgorithms;

template < class DataTypes >
class EdgeSetGeometryAlgorithms;

template <class DataTypes>
class EdgeSetTopologyLoader;

class EdgesAdded;
class EdgesRemoved;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;

typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::VertexEdges VertexEdges;


/////////////////////////////////////////////////////////
/// EdgeSetTopology objects
/////////////////////////////////////////////////////////


/** Describes a topological object that consists as a set of points and lines connecting these points */
template<class DataTypes>
class EdgeSetTopology : public PointSetTopology <DataTypes>
{
public:
    EdgeSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~EdgeSetTopology() {}

    virtual void init();

    /** \brief Returns the EdgeSetTopologyContainer object of this EdgeSetTopology.
    */
    EdgeSetTopologyContainer *getEdgeSetTopologyContainer() const
    {
        return static_cast<EdgeSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the EdgeSetTopologyModifier object of this EdgeSetTopology.
    */
    EdgeSetTopologyModifier<DataTypes> *getEdgeSetTopologyModifier() const
    {
        return static_cast<EdgeSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the EdgeSetTopologyAlgorithms object of this EdgeSetTopology.
    */
    EdgeSetTopologyAlgorithms<DataTypes> *getEdgeSetTopologyAlgorithms() const
    {
        return static_cast<EdgeSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the EdgeSetGeometryAlgorithms object of this EdgeSetTopology.
    */
    EdgeSetGeometryAlgorithms<DataTypes> *getEdgeSetGeometryAlgorithms() const
    {
        return static_cast<EdgeSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
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
