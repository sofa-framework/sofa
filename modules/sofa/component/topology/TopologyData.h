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
    PointDataImpl(InputIterator first, InputIterator last): TopologyDataImpl<Point, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    PointDataImpl(typename PointDataImpl<VecT>::const_iterator first, typename PointDataImpl<VecT>::const_iterator last): TopologyDataImpl<Point, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

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
    EdgeDataImpl(InputIterator first, InputIterator last): TopologyDataImpl<Edge, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    EdgeDataImpl(typename EdgeDataImpl<VecT>::const_iterator first, typename EdgeDataImpl<VecT>::const_iterator last): TopologyDataImpl<Edge, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

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
    TriangleDataImpl(InputIterator first, InputIterator last): TopologyDataImpl<Triangle, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TriangleDataImpl(typename TriangleDataImpl<VecT>::const_iterator first, typename TriangleDataImpl<VecT>::const_iterator last): TopologyDataImpl<Triangle, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

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
    QuadDataImpl(InputIterator first, InputIterator last): TopologyDataImpl<Quad, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    QuadDataImpl(typename QuadDataImpl<VecT>::const_iterator first, typename QuadDataImpl<VecT>::const_iterator last): TopologyDataImpl<Quad, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

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
    TetrahedronDataImpl(InputIterator first, InputIterator last): TopologyDataImpl<Tetrahedron, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TetrahedronDataImpl(typename TetrahedronDataImpl<VecT>::const_iterator first, typename TetrahedronDataImpl<VecT>::const_iterator last): TopologyDataImpl<Tetrahedron, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

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
    HexahedronDataImpl(InputIterator first, InputIterator last): TopologyDataImpl<Hexahedron, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    HexahedronDataImpl(typename HexahedronDataImpl<VecT>::const_iterator first, typename HexahedronDataImpl<VecT>::const_iterator last): TopologyDataImpl<Hexahedron, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

protected:
    HexahedronSetTopologyEngine<VecT>* m_topologicalEngine;

};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
