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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATAHANDLER_H
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATAHANDLER_H

#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/component/topology/TopologyData.h>


namespace sofa
{

namespace component
{

namespace topology
{
// Define topology elements
using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::Point Point;
typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::Triangle Triangle;
typedef BaseMeshTopology::Quad Quad;
typedef BaseMeshTopology::Tetrahedron Tetrahedron;
typedef BaseMeshTopology::Hexahedron Hexahedron;

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/
template< class TopologyElementType, class VecT>
class TopologyDataHandlerImpl : public sofa::core::topology::TopologyHandler
{
public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    /// size_type
    typedef typename container_type::size_type size_type;
    /// reference to a value (read-write)
    typedef typename container_type::reference reference;
    /// const reference to a value (read only)
    typedef typename container_type::const_reference const_reference;
    /// const iterator
    typedef typename container_type::const_iterator const_iterator;


    /** Public fonction to apply creation and destruction functions */
public:

    // constructor
    TopologyDataHandlerImpl(): sofa::core::topology::TopologyDataImpl(), m_topologyData(NULL) {}

    /// Apply removing current elementType elements
    virtual void applyDestroyFunction(unsigned int, value_type& /*t*/) {/*t = VecT();*/}
    /// Apply adding current elementType elements
    virtual void applyCreateFunction(unsigned int, value_type&,
            const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double > &) {}

    /// WARNING NEEED TO UNIFY THIS
    /// Apply adding current elementType elements
    virtual void applyCreateFunction(unsigned int, value_type&, const TopologyElementType& ,
            const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double > &) {}

    ///////////////////////// Functions on Points //////////////////////////////////////
    /// Apply adding points elements.
    virtual void applyPointCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing points elements.
    virtual void applyPointDestruction(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply swap between point indices elements.
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
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing edges elements.
    virtual void applyEdgeDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between edges indices elements.
    virtual void applyEdgeIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on edges elements.
    virtual void applyeEdgeRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved edges elements.
    virtual void applyEdgeMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/) {}
    /// Apply removing function on moved edges elements.
    virtual void applyEdgeMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Triangles //////////////////////////////////////
    /// Apply adding triangles elements.
    virtual void applyTriangleCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing triangles elements.
    virtual void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between triangles indices elements.
    virtual void applyTriangleIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on triangles elements.
    virtual void applyeTriangleRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved triangles elements.
    virtual void applyTriangleMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/) {}
    /// Apply removing function on moved triangles elements.
    virtual void applyTriangleMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Quads //////////////////////////////////////
    /// Apply adding quads elements.
    virtual void applyQuadCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing quads elements.
    virtual void applyQuadDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between quads indices elements.
    virtual void applyQuadIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on quads elements.
    virtual void applyeQuadRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved quads elements.
    virtual void applyQuadMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/) {}
    /// Apply removing function on moved quads elements.
    virtual void applyQuadMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Tetrahedron //////////////////////////////////////
    /// Apply adding tetrahedron elements.
    virtual void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing tetrahedron elements.
    virtual void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between tetrahedron indices elements.
    virtual void applyTetrahedronIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on tetrahedron elements.
    virtual void applyeTetrahedronRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved tetrahedron elements.
    virtual void applyTetrahedronMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/) {}
    /// Apply removing function on moved tetrahedron elements.
    virtual void applyTetrahedronMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    ///////////////////////// Functions on Hexahedron //////////////////////////////////////
    /// Apply adding hexahedron elements.
    virtual void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing hexahedron elements.
    virtual void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}
    /// Apply swap between hexahedron indices elements.
    virtual void applyHexahedronIndicesSwap(unsigned int /*i1*/, unsigned int /*i2*/ ) {}
    /// Apply renumbering on hexahedron elements.
    virtual void applyeHexahedronRenumbering(const sofa::helper::vector<unsigned int>& /*indices*/) {}
    /// Apply adding function on moved hexahedron elements.
    virtual void applyHexahedronMovedCreation(const sofa::helper::vector<unsigned int>& /*indexList*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/) {}
    /// Apply removing function on moved hexahedron elements.
    virtual void applyHexahedronMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    /// Handle EdgeSetTopology related events, ignore others. DEPRECATED
    virtual void handleTopologyEvents( std::list< const core::topology::TopologyChange *>::const_iterator changeIt,
            std::list< const core::topology::TopologyChange *>::const_iterator &end );

protected:
    /// Swaps values at indices i1 and i2.
    void swap( unsigned int i1, unsigned int i2 );

    /// Add some values. Values are added at the end of the vector.
    void add( unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    void add( unsigned int nbElements,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    /// Remove the values corresponding to the Edges removed.
    void remove( const sofa::helper::vector<unsigned int> &index );

    /// Reorder the values.
    void renumber( const sofa::helper::vector<unsigned int> &index );

    /// Move a list of points
    void move( const sofa::helper::vector<unsigned int> &indexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

protected:
    sofa::component::topology::TopologyDataImpl<TopologyElementType, VecT>* m_topologyData;
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class PointDataHandlerImpl : public TopologyDataHandlerImpl<Point, VecT>
{
public:
    typedef typename TopologyDataHandlerImpl<Point, VecT>::container_type container_type;
    typedef typename TopologyDataHandlerImpl<Point, VecT>::value_type value_type;

public:
    PointDataHandlerImpl() : TopologyDataHandlerImpl () {}

    /// Apply adding point elements.
    void applyPointCreation(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing point elements.
    void applyPointDestruction(const sofa::helper::vector<unsigned int> & indices);
    /// Apply removing points elements.
    void applyPointIndicesSwap(unsigned int i1, unsigned int i2 );
    /// Apply removing points elements.
    void applyPointRenumbering(const sofa::helper::vector<unsigned int>& indices);
    /// Apply removing points elements.
    void applyPointMove(const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< Point >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class EdgeDataHandlerImpl : public TopologyDataHandlerImpl<Edge, VecT>
{
public:
    typedef typename TopologyDataHandlerImpl<Edge, VecT>::container_type container_type;
    typedef typename TopologyDataHandlerImpl<Edge, VecT>::value_type value_type;

public:
    EdgeDataHandlerImpl() : TopologyDataHandlerImpl () {}

    /// Apply adding Edge elements.
    void applyEdgeCreation(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< Edge >& elems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing Edge elements.
    void applyEdgeDestruction(const sofa::helper::vector<unsigned int> & indices);
    /// Apply swap between Edge indices elements.
    virtual void applyEdgeIndicesSwap(unsigned int i1, unsigned int i2 );
    /// Apply renumbering on Edge elements.
    virtual void applyeEdgeRenumbering(const sofa::helper::vector<unsigned int>& indices);
    /// Apply adding function on moved Edge elements.
    virtual void applyEdgeMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< Edge >& elems);
    /// Apply removing function on moved Edge elements.
    virtual void applyEdgeMovedDestruction(const sofa::helper::vector<unsigned int> & indices);
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Triangle Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TriangleDataHandlerImpl : public TopologyDataHandlerImpl<Triangle, VecT>
{
public:
    typedef typename TopologyDataHandlerImpl<Triangle, VecT>::container_type container_type;
    typedef typename TopologyDataHandlerImpl<Triangle, VecT>::value_type value_type;

public:
    TriangleDataHandlerImpl() : TopologyDataHandlerImpl () {}

    /// Apply adding Triangle elements.
    void applyTriangleCreation(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< Triangle >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing Triangle elements.
    void applyTriangleDestruction(const sofa::helper::vector<unsigned int> & indices);
    /// Apply swap between Triangle indices elements.
    virtual void applyTriangleIndicesSwap(unsigned int i1, unsigned int i2 );
    /// Apply renumbering on Triangle elements.
    virtual void applyeTriangleRenumbering(const sofa::helper::vector<unsigned int>& indices);
    /// Apply adding function on moved Triangle elements.
    virtual void applyTriangleMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< Triangle >& /*elems*/);
    /// Apply removing function on moved Triangle elements.
    virtual void applyTriangleMovedDestruction(const sofa::helper::vector<unsigned int> & indices);
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Quad Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class QuadDataHandlerImpl : public TopologyDataHandlerImpl<Quad, VecT>
{
public:
    typedef typename TopologyDataHandlerImpl<Quad, VecT>::container_type container_type;
    typedef typename TopologyDataHandlerImpl<Quad, VecT>::value_type value_type;

public:
    QuadDataHandlerImpl() : TopologyDataHandlerImpl () {}

    /// Apply adding Quad elements.
    void applyQuadCreation(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< Quad >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing Quad elements.
    void applyQuadDestruction(const sofa::helper::vector<unsigned int> & indices);
    /// Apply swap between Quad indices elements.
    virtual void applyQuadIndicesSwHexahedronap(unsigned int i1, unsigned int i2 );
    /// Apply renumbering on Quad elements.
    virtual void applyeQuadRenumbering(const sofa::helper::vector<unsigned int>& indices);
    /// Apply adding function on moved Quad elements.
    virtual void applyQuadMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< Quad >& /*elems*/);
    /// Apply removing function on moved Quad elements.
    virtual void applyQuadMovedDestruction(const sofa::helper::vector<unsigned int> & indices);
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TetrahedronDataHandlerImpl : public TopologyDataHandlerImpl<Tetrahedron, VecT>
{
public:
    typedef typename TopologyDataHandlerImpl<Tetrahedron, VecT>::container_type container_type;
    typedef typename TopologyDataHandlerImpl<Tetrahedron, VecT>::value_type value_type;

public:
    TetrahedronDataHandlerImpl() : TopologyDataHandlerImpl () {}

    /// Apply adding Tetrahedron elements.
    void applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< Tetrahedron >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing Tetrahedron elements.
    void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & indices);
    /// Apply swap between Tetrahedron indices elements.
    virtual void applyTetrahedronIndicesSwap(unsigned int i1, unsigned int i2 );
    /// Apply renumbering on Tetrahedron elements.
    virtual void applyeTetrahedronRenumbering(const sofa::helper::vector<unsigned int>& indices);
    /// Apply adding function on moved Tetrahedron elements.
    virtual void applyTetrahedronMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< Tetrahedron >& /*elems*/);
    /// Apply removing function on moved Tetrahedron elements.
    virtual void applyTetrahedronMovedDestruction(const sofa::helper::vector<unsigned int> & indices);
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Hexahedron Topology Data Implementation   //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class HexahedronDataHandlerImpl : public TopologyDataHandlerImpl<Hexahedron, VecT>
{
public:
    typedef typename TopologyDataHandlerImpl<Hexahedron, VecT>::container_type container_type;
    typedef typename TopologyDataHandlerImpl<Hexahedron, VecT>::value_type value_type;

public:
    HexahedronDataHandlerImpl() : TopologyDataHandlerImpl () {}

    /// Apply adding Hexahedron elements.
    void applyHexahedronCreation(const sofa::helper::vector< unsigned int >& indices,
            const sofa::helper::vector< Hexahedron >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing Hexahedron elements.
    void applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & indices);
    /// Apply swap between Hexahedron indices elements.
    virtual void applyHexahedronIndicesSwap(unsigned int i1, unsigned int i2 );
    /// Apply renumbering on Hexahedron elements.
    virtual void applyeHexahedronRenumbering(const sofa::helper::vector<unsigned int>& indices);
    /// Apply adding function on moved Hexahedron elements.
    virtual void applyHexahedronMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< Hexahedron >& /*elems*/);
    /// Apply removing function on moved Hexahedron elements.
    virtual void applyHexahedronMovedDestruction(const sofa::helper::vector<unsigned int> & indices);
};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATAHANDLER_H
