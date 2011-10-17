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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/component/component.h>

#include <sofa/component/topology/PointSetTopologyEngine.h>
#include <sofa/component/topology/EdgeSetTopologyEngine.h>
#include <sofa/component/topology/TriangleSetTopologyEngine.h>
#include <sofa/component/topology/QuadSetTopologyEngine.h>
#include <sofa/component/topology/TetrahedronSetTopologyEngine.h>
#include <sofa/component/topology/HexahedronSetTopologyEngine.h>


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
class TopologyDataImpl : public sofa::core::topology::BaseTopologyData<VecT>
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


    /// Constructors
public:
    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TopologyDataImpl( )
        : sofa::core::topology::BaseTopologyData< VecT >(0, false, false),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    TopologyDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : sofa::core::topology::BaseTopologyData< VecT >(data),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    TopologyDataImpl(size_type n, const value_type& value) : sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n, value);
        this->endEdit();
    }
    /// Constructor
    explicit TopologyDataImpl(size_type n): sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n);
        this->endEdit();
    }
    /// Constructor
    TopologyDataImpl(const container_type& x): sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        (*data) = x;
        this->endEdit();
    }

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TopologyDataImpl(InputIterator first, InputIterator last): sofa::core::topology::BaseTopologyData< container_type >(0, false, false)
    {
        container_type* data = this->beginEdit();
        data->assign(first, last);
        this->endEdit();
    }
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TopologyDataImpl(const_iterator first, const_iterator last): sofa::core::topology::BaseTopologyData< container_type >(0, false, false)
    {
        container_type* data = this->beginEdit();
        data->assign(first, last);
        this->endEdit();
    }
#endif /* __STL_MEMBER_TEMPLATES */


    /** Public fonction to apply creation and destruction functions */
public:
    /// Apply removing current elementType elements
    virtual void applyDestroyFunction(unsigned int, value_type& /*t*/) {/*t = VecT();*/}
    /// Apply adding current elementType elements
    virtual void applyCreateFunction(unsigned int, value_type&,
            const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double > &) {}


    ///////////////////////// Functions on Points //////////////////////////////////////
    /// Apply adding points elements.
    virtual void applyPointCreation(const sofa::helper::vector< unsigned int >& /*indices*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    /// Apply removing function on moved hexahedron elements.
    virtual void applyHexahedronMovedDestruction(const sofa::helper::vector<unsigned int> & /*indices*/) {}



    /// Handle EdgeSetTopology related events, ignore others. DEPRECATED
    void handleTopologyEvents( std::list< const core::topology::TopologyChange *>::const_iterator changeIt,
            std::list< const core::topology::TopologyChange *>::const_iterator &end );

    /// Handle other event on topology changes
    virtual void applyTopologyChangesFunction();


    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    virtual void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* ) {}

    /// Allow to add additionnal dependencies to others Data.
    void addInputData(sofa::core::objectmodel::BaseData* _data);

    /// Function to link the topological Data with the engine and the current topology. And init everything.
    /// This function should be used at the end of the all declaration link to this Data while using it in a component.
    void registerTopologicalData();


    value_type& operator[](int i)
    {
        container_type& data = *(this->beginEdit());
        value_type& result = data[i];
        this->endEdit();
        return result;
    }


    /// Link Data to topology arrays
    void linkToPointDataArray();
    void linkToEdgeDataArray();
    void linkToTriangleDataArray();
    void linkToQuadDataArray();
    void linkToTetrahedronDataArray();
    void linkToHexahedronDataArray();

protected:
    /// Swaps values at indices i1 and i2.
    void swap( unsigned int i1, unsigned int i2 );

    /// Add some values. Values are added at the end of the vector.
    void add( unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elems,
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
    sofa::core::topology::TopologyEngine* m_topologicalEngine;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class PointDataImpl : public TopologyDataImpl<Point, VecT>
{
public:
    typedef typename TopologyDataImpl<Point, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<Point, VecT>::value_type value_type;

    PointDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<Point, VecT>(data)
        , m_topologicalEngine(NULL)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    PointDataImpl() : TopologyDataImpl<Point, VecT>()
        , m_topologicalEngine(NULL)
    {}

    /// Constructor
    PointDataImpl(typename TopologyDataImpl<Point, VecT>::size_type n, const value_type& value): TopologyDataImpl<Point, VecT>(n,value) {}

    /// Constructor
    explicit PointDataImpl(typename TopologyDataImpl<Point, VecT>::size_type n): TopologyDataImpl<Point, VecT>(n) {}
    /// Constructor
    PointDataImpl(const container_type& x): TopologyDataImpl<Point, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    PointDataImpl(InputIterator first, InputIterator last): TopologyData<Point, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    PointDataImpl(typename PointDataImpl<VecT>::const_iterator first, typename PointDataImpl<VecT>::const_iterator last): TopologyData<Point, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

public:
    /// Apply adding point elements.
    void applyPointCreation(unsigned int nbPoints,
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);


protected:
    PointSetTopologyEngine<VecT>* m_topologicalEngine;

};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class EdgeDataImpl : public TopologyDataImpl<Edge, VecT>
{
public:
    typedef typename TopologyDataImpl<Edge, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<Edge, VecT>::value_type value_type;

    EdgeDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<Edge, VecT>(data)
        , m_topologicalEngine(NULL)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    EdgeDataImpl() : TopologyDataImpl<Edge, VecT>()
        , m_topologicalEngine(NULL)
    {}

    /// Constructor
    EdgeDataImpl(typename TopologyDataImpl<Edge, VecT>::size_type n, const value_type& value): TopologyDataImpl<Edge, VecT>(n,value) {}

    /// Constructor
    explicit EdgeDataImpl(typename TopologyDataImpl<Edge, VecT>::size_type n): TopologyDataImpl<Edge, VecT>(n) {}
    /// Constructor
    EdgeDataImpl(const container_type& x): TopologyDataImpl<Edge, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    EdgeDataImpl(InputIterator first, InputIterator last): TopologyData<Edge, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    EdgeDataImpl(typename EdgeDataImpl<VecT>::const_iterator first, typename EdgeDataImpl<VecT>::const_iterator last): TopologyData<Edge, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

    /// Apply adding Edge elements.
    void applyEdgeCreation(unsigned int nbEdges,
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing function on moved Edge elements.
    virtual void applyEdgeMovedDestruction(const sofa::helper::vector<unsigned int> & indices);


protected:
    EdgeSetTopologyEngine<VecT>* m_topologicalEngine;

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Triangle Topology Data Implementation   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TriangleDataImpl : public TopologyDataImpl<Triangle, VecT>
{
public:
    typedef typename TopologyDataImpl<Triangle, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<Triangle, VecT>::value_type value_type;

    TriangleDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<Triangle, VecT>(data)
        , m_topologicalEngine(NULL)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TriangleDataImpl() : TopologyDataImpl<Triangle, VecT>()
        , m_topologicalEngine(NULL)
    {}

    /// Constructor
    TriangleDataImpl(typename TopologyDataImpl<Triangle, VecT>::size_type n, const value_type& value): TopologyDataImpl<Triangle, VecT>(n,value) {}

    /// Constructor
    explicit TriangleDataImpl(typename TopologyDataImpl<Triangle, VecT>::size_type n): TopologyDataImpl<Triangle, VecT>(n) {}
    /// Constructor
    TriangleDataImpl(const container_type& x): TopologyDataImpl<Triangle, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TriangleDataImpl(InputIterator first, InputIterator last): TopologyData<Triangle, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TriangleDataImpl(typename TriangleDataImpl<VecT>::const_iterator first, typename TriangleDataImpl<VecT>::const_iterator last): TopologyData<Triangle, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

    /// Apply adding Triangle elements.
    void applyTriangleCreation(unsigned int nbTriangles,
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing function on moved Triangle elements.
    virtual void applyTriangleMovedDestruction(const sofa::helper::vector<unsigned int> & indices);


protected:
    TriangleSetTopologyEngine<VecT>* m_topologicalEngine;

};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Quad Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class QuadDataImpl : public TopologyDataImpl<Quad, VecT>
{
public:
    typedef typename TopologyDataImpl<Quad, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<Quad, VecT>::value_type value_type;

    QuadDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<Quad, VecT>(data)
        , m_topologicalEngine(NULL)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    QuadDataImpl() : TopologyDataImpl<Quad, VecT>()
        , m_topologicalEngine(NULL)
    {}

    /// Constructor
    QuadDataImpl(typename TopologyDataImpl<Quad, VecT>::size_type n, const value_type& value): TopologyDataImpl<Quad, VecT>(n,value) {}

    /// Constructor
    explicit QuadDataImpl(typename TopologyDataImpl<Quad, VecT>::size_type n): TopologyDataImpl<Quad, VecT>(n) {}
    /// Constructor
    QuadDataImpl(const container_type& x): TopologyDataImpl<Quad, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    QuadDataImpl(InputIterator first, InputIterator last): TopologyData<Quad, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    QuadDataImpl(typename QuadDataImpl<VecT>::const_iterator first, typename QuadDataImpl<VecT>::const_iterator last): TopologyData<Quad, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

    /// Apply adding Quad elements.
    void applyQuadCreation(unsigned int nbQuads,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing Quad elements.
    void applyQuadDestruction(const sofa::helper::vector<unsigned int> & indices);
    /// Apply swap between Quad indices elements.
    virtual void applyQuadIndicesSwap(unsigned int i1, unsigned int i2 );
    /// Apply renumbering on Quad elements.
    virtual void applyeQuadRenumbering(const sofa::helper::vector<unsigned int>& indices);
    /// Apply adding function on moved Quad elements.
    virtual void applyQuadMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing function on moved Quad elements.
    virtual void applyQuadMovedDestruction(const sofa::helper::vector<unsigned int> & indices);


protected:
    QuadSetTopologyEngine<VecT>* m_topologicalEngine;

};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TetrahedronDataImpl : public TopologyDataImpl<Tetrahedron, VecT>
{
public:
    typedef typename TopologyDataImpl<Tetrahedron, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<Tetrahedron, VecT>::value_type value_type;

    TetrahedronDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<Tetrahedron, VecT>(data)
        , m_topologicalEngine(NULL)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TetrahedronDataImpl() : TopologyDataImpl<Tetrahedron, VecT>()
        , m_topologicalEngine(NULL)
    {}

    /// Constructor
    TetrahedronDataImpl(typename TopologyDataImpl<Tetrahedron, VecT>::size_type n, const value_type& value): TopologyDataImpl<Tetrahedron, VecT>(n,value) {}

    /// Constructor
    explicit TetrahedronDataImpl(typename TopologyDataImpl<Tetrahedron, VecT>::size_type n): TopologyDataImpl<Tetrahedron, VecT>(n) {}
    /// Constructor
    TetrahedronDataImpl(const container_type& x): TopologyDataImpl<Tetrahedron, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TetrahedronDataImpl(InputIterator first, InputIterator last): TopologyData<Tetrahedron, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TetrahedronDataImpl(typename TetrahedronDataImpl<VecT>::const_iterator first, typename TetrahedronDataImpl<VecT>::const_iterator last): TopologyData<Tetrahedron, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

    /// Apply adding Tetrahedron elements.
    void applyTetrahedronCreation(unsigned int nbTetrahedra,
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing function on moved Tetrahedron elements.
    virtual void applyTetrahedronMovedDestruction(const sofa::helper::vector<unsigned int> & indices);


protected:
    TetrahedronSetTopologyEngine<VecT>* m_topologicalEngine;

};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Hexahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class HexahedronDataImpl : public TopologyDataImpl<Hexahedron, VecT>
{
public:
    typedef typename TopologyDataImpl<Hexahedron, VecT>::container_type container_type;
    typedef typename TopologyDataImpl<Hexahedron, VecT>::value_type value_type;

    HexahedronDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologyDataImpl<Hexahedron, VecT>(data)
        , m_topologicalEngine(NULL)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    HexahedronDataImpl() : TopologyDataImpl<Hexahedron, VecT>()
        , m_topologicalEngine(NULL)
    {}

    /// Constructor
    HexahedronDataImpl(typename TopologyDataImpl<Hexahedron, VecT>::size_type n, const value_type& value): TopologyDataImpl<Hexahedron, VecT>(n,value) {}

    /// Constructor
    explicit HexahedronDataImpl(typename TopologyDataImpl<Hexahedron, VecT>::size_type n): TopologyDataImpl<Hexahedron, VecT>(n) {}
    /// Constructor
    HexahedronDataImpl(const container_type& x): TopologyDataImpl<Hexahedron, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    HexahedronDataImpl(InputIterator first, InputIterator last): TopologyData<Hexahedron, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    HexahedronDataImpl(typename HexahedronDataImpl<VecT>::const_iterator first, typename HexahedronDataImpl<VecT>::const_iterator last): TopologyData<Hexahedron, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

    /// Apply adding Hexahedron elements.
    void applyHexahedronCreation(unsigned int nbHexahedra,
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing function on moved Hexahedron elements.
    virtual void applyHexahedronMovedDestruction(const sofa::helper::vector<unsigned int> & indices);


protected:
    HexahedronSetTopologyEngine<VecT>* m_topologicalEngine;

};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
