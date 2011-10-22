/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGYHANDLER_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGYHANDLER_H

#include <sofa/core/topology/TopologyChange.h>

namespace sofa
{

namespace core
{

namespace topology
{

using namespace sofa::helper;

typedef Topology::Point            Point;
typedef Topology::Edge             Edge;
typedef Topology::Triangle         Triangle;
typedef Topology::Quad             Quad;
typedef Topology::Tetrahedron      Tetrahedron;
typedef Topology::Hexahedron       Hexahedron;


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Generic Handling of Topology Event    /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SOFA_CORE_API TopologyHandler
{
public:
    TopologyHandler() {}

    virtual ~TopologyHandler() {}

    virtual void ApplyTopologyChanges(const std::list< const core::topology::TopologyChange *>& _topologyChangeEvents, const unsigned int _dataSize);
    /// Handle EdgeSetTopology related events, ignore others. DEPRECATED
    /*virtual void handleTopologyEvents( std::list< const core::topology::TopologyChange *>::const_iterator changeIt,
                                       std::list< const core::topology::TopologyChange *>::const_iterator &end,
                                       const unsigned int totalPointSetArraySize = 0){}
    */
    /// Apply adding points elements.
    virtual void applyPointCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< Point >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing points elements.
    virtual void applyPointDestruction(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply swap between point indicPointes elements.
    virtual void applyPointIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on points elements.
    virtual void applyPointRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply moving points elements.
    virtual void applyPointMove(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}



    ///////////////////////// Functions on Edges //////////////////////////////////////
    /// Apply adding edges elements.
    virtual void applyEdgeCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< Edge >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing edges elements.
    virtual void applyEdgeDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between edges indices elements.
    virtual void applyEdgeIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on edges elements.
    virtual void applyEdgeRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved edges elements.
    virtual void applyEdgeMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< Edge >& /*elems*/) {}
    /// Apply removing function on moved edges elements.
    virtual void applyEdgeMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Triangles //////////////////////////////////////
    /// Apply adding triangles elements.
    virtual void applyTriangleCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< Triangle >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing triangles elements.
    virtual void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between triangles indices elements.
    virtual void applyTriangleIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on triangles elements.
    virtual void applyTriangleRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved triangles elements.
    virtual void applyTriangleMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< Triangle >& /*elems*/) {}
    /// Apply removing function on moved triangles elements.
    virtual void applyTriangleMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Quads //////////////////////////////////////
    /// Apply adding quads elements.
    virtual void applyQuadCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< Quad >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing quads elements.
    virtual void applyQuadDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between quads indices elements.
    virtual void applyQuadIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on quads elements.
    virtual void applyQuadRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved quads elements.
    virtual void applyQuadMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< Quad >& /*elems*/) {}
    /// Apply removing function on moved quads elements.
    virtual void applyQuadMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Tetrahedron //////////////////////////////////////
    /// Apply adding tetrahedron elements.
    virtual void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< Tetrahedron >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing tetrahedron elements.
    virtual void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between tetrahedron indices elements.
    virtual void applyTetrahedronIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on tetrahedron elements.
    virtual void applyTetrahedronRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved tetrahedron elements.
    virtual void applyTetrahedronMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< Tetrahedron >& /*elems*/) {}
    /// Apply removing function on moved tetrahedron elements.
    virtual void applyTetrahedronMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Hexahedron //////////////////////////////////////
    /// Apply adding hexahedron elements.
    virtual void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< Hexahedron >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing hexahedron elements.
    virtual void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between hexahedron indices elements.
    virtual void applyHexahedronIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on hexahedron elements.
    virtual void applyHexahedronRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved hexahedron elements.
    virtual void applyHexahedronMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< Hexahedron >& /*elems*/) {}
    /// Apply removing function on moved hexahedron elements.
    virtual void applyHexahedronMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    virtual bool isTopologyDataRegistered() {return false;}

    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int /*i1*/, unsigned int /*i2*/ ) {}

    /// Reorder the values.
    virtual void renumber( const sofa::helper::vector<unsigned int> &/*index*/ ) {}

protected:
    /// to handle PointSubsetData
    void setDataSetArraySize(const unsigned int s) { lastElementIndex = s-1; }

    /// to handle properly the removal of items, the container must know the index of the last element
    unsigned int lastElementIndex;
};


} // namespace topology

} // namespace core

} // namespace sofa


#endif // SOFA_CORE_TOPOLOGY_TOPOLOGYHANDLER_H
