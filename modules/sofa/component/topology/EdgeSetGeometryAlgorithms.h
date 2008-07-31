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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_H

#include <sofa/component/topology/PointSetGeometryAlgorithms.h>

#include <sofa/component/topology/EdgeSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class EdgeSetTopologyContainer;

/** \brief A class used as an interface with an array : Useful to compute geometric information on each edge in an efficient way
*
*/
template < class T>
class BasicArrayInterface
{
public:
    // Access to i-th element.
    virtual T & operator[](int i)=0;
    virtual ~BasicArrayInterface() {}

};

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;
typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::VertexEdges VertexEdges;

/**
* A class that provides geometry information on an EdgeSet.
*/
template < class DataTypes >
class EdgeSetGeometryAlgorithms : public PointSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


    EdgeSetGeometryAlgorithms()
        : PointSetGeometryAlgorithms<DataTypes>()
    {}

    virtual ~EdgeSetGeometryAlgorithms() {}

    virtual void init();

    /// computes the length of edge no i and returns it
    virtual Real computeEdgeLength(const unsigned int i) const;

    /// computes the edge length of all edges are store in the array interface
    virtual void computeEdgeLength( BasicArrayInterface<Real> &ai) const;		// TODO: clarify, why not to use a vector here

    /// computes the initial length of edge no i and returns it
    virtual Real computeRestEdgeLength(const unsigned int i) const;

    /// computes the initial square length of edge no i and returns it
    virtual Real computeRestSquareEdgeLength(const unsigned int i) const;

private:
    EdgeSetTopologyContainer* m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
