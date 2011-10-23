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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSUBSETDATA_H
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSUBSETDATA_H

#include <sofa/helper/vector.h>
#include <sofa/component/component.h>

#include <sofa/core/topology/BaseTopologyData.h>
#include <sofa/component/topology/TopologyEngine.h>
#include <sofa/component/topology/TopologySubsetDataHandler.h>

namespace sofa
{

namespace component
{

namespace topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/
template< class TopologyElementType, class VecT>
class TopologySubsetDataImpl : public sofa::core::topology::BaseTopologyData<VecT>
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
    TopologySubsetDataImpl( )
        : sofa::core::topology::BaseTopologyData< VecT >(0, false, false),
          m_topologicalEngine(NULL),
          m_topologyHandler(NULL)
    {}

    /// Constructor
    TopologySubsetDataImpl( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : sofa::core::topology::BaseTopologyData< VecT >(data),
          m_topologicalEngine(NULL),
          m_topologyHandler(NULL)
    {}

    /// Constructor
    TopologySubsetDataImpl(size_type n, const value_type& value) : sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n, value);
        this->endEdit();
    }
    /// Constructor
    explicit TopologySubsetDataImpl(size_type n): sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n);
        this->endEdit();
    }
    /// Constructor
    TopologySubsetDataImpl(const container_type& x): sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        (*data) = x;
        this->endEdit();
    }

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TopologySubsetDataImpl(InputIterator first, InputIterator last): sofa::core::topology::BaseTopologyData< container_type >(0, false, false)
    {
        container_type* data = this->beginEdit();
        data->assign(first, last);
        this->endEdit();
    }
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TopologySubsetDataImpl(const_iterator first, const_iterator last): sofa::core::topology::BaseTopologyData< container_type >(0, false, false)
    {
        container_type* data = this->beginEdit();
        data->assign(first, last);
        this->endEdit();
    }
#endif /* __STL_MEMBER_TEMPLATES */

    ~TopologySubsetDataImpl();

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    virtual void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology, sofa::core::topology::TopologyHandler* _topologyHandler);

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    virtual void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

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
    virtual void linkToElementDataArray() {}

    virtual void createTopologyHandler() {}

    sofa::component::topology::TopologyEngineImpl<VecT>* m_topologicalEngine;
    sofa::component::topology::TopologySubsetDataHandler<TopologyElementType,VecT>* m_topologyHandler;
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class PointSubsetData : public TopologySubsetDataImpl<Point, VecT>
{
public:
    typedef typename TopologySubsetDataImpl<Point, VecT>::container_type container_type;
    typedef typename TopologySubsetDataImpl<Point, VecT>::value_type value_type;

    PointSubsetData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologySubsetDataImpl<Point, VecT>(data)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    PointSubsetData() : TopologySubsetDataImpl<Point, VecT>()
    {}

    /// Constructor
    PointSubsetData(typename TopologySubsetDataImpl<Point, VecT>::size_type n, const value_type& value): TopologySubsetDataImpl<Point, VecT>(n,value) {}

    /// Constructor
    explicit PointSubsetData(typename TopologySubsetDataImpl<Point, VecT>::size_type n): TopologySubsetDataImpl<Point, VecT>(n) {}
    /// Constructor
    PointSubsetData(const container_type& x): TopologySubsetDataImpl<Point, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    PointSubsetData(InputIterator first, InputIterator last): TopologySubsetDataImpl<Point, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    PointSubsetData(typename PointSubsetData<VecT>::const_iterator first, typename PointSubsetData<VecT>::const_iterator last): TopologySubsetDataImpl<Point, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

protected:
    void linkToElementDataArray() {this->linkToPointDataArray();}
};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class EdgeSubsetData : public TopologySubsetDataImpl<Edge, VecT>
{
public:
    typedef typename TopologySubsetDataImpl<Edge, VecT>::container_type container_type;
    typedef typename TopologySubsetDataImpl<Edge, VecT>::value_type value_type;

    EdgeSubsetData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologySubsetDataImpl<Edge, VecT>(data)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    EdgeSubsetData() : TopologySubsetDataImpl<Edge, VecT>()
    {}

    /// Constructor
    EdgeSubsetData(typename TopologySubsetDataImpl<Edge, VecT>::size_type n, const value_type& value): TopologySubsetDataImpl<Edge, VecT>(n,value) {}

    /// Constructor
    explicit EdgeSubsetData(typename TopologySubsetDataImpl<Edge, VecT>::size_type n): TopologySubsetDataImpl<Edge, VecT>(n) {}
    /// Constructor
    EdgeSubsetData(const container_type& x): TopologySubsetDataImpl<Edge, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    EdgeSubsetData(InputIterator first, InputIterator last): TopologySubsetDataImpl<Edge, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    EdgeSubsetData(typename EdgeSubsetData<VecT>::const_iterator first, typename EdgeSubsetData<VecT>::const_iterator last): TopologySubsetDataImpl<Edge, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

protected:
    void linkToElementDataArray() {this->linkToEdgeDataArray();}

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Triangle Topology Data Implementation   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TriangleSubsetData : public TopologySubsetDataImpl<Triangle, VecT>
{
public:
    typedef typename TopologySubsetDataImpl<Triangle, VecT>::container_type container_type;
    typedef typename TopologySubsetDataImpl<Triangle, VecT>::value_type value_type;

    TriangleSubsetData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologySubsetDataImpl<Triangle, VecT>(data)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TriangleSubsetData() : TopologySubsetDataImpl<Triangle, VecT>()
    {}

    /// Constructor
    TriangleSubsetData(typename TopologySubsetDataImpl<Triangle, VecT>::size_type n, const value_type& value): TopologySubsetDataImpl<Triangle, VecT>(n,value) {}

    /// Constructor
    explicit TriangleSubsetData(typename TopologySubsetDataImpl<Triangle, VecT>::size_type n): TopologySubsetDataImpl<Triangle, VecT>(n) {}
    /// Constructor
    TriangleSubsetData(const container_type& x): TopologySubsetDataImpl<Triangle, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TriangleSubsetData(InputIterator first, InputIterator last): TopologySubsetDataImpl<Triangle, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TriangleSubsetData(typename TriangleSubsetData<VecT>::const_iterator first, typename TriangleSubsetData<VecT>::const_iterator last): TopologySubsetDataImpl<Triangle, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

protected:
    void linkToElementDataArray() {this->linkToTriangleDataArray();}

};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Quad Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class QuadSubsetData : public TopologySubsetDataImpl<Quad, VecT>
{
public:
    typedef typename TopologySubsetDataImpl<Quad, VecT>::container_type container_type;
    typedef typename TopologySubsetDataImpl<Quad, VecT>::value_type value_type;

    QuadSubsetData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologySubsetDataImpl<Quad, VecT>(data)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    QuadSubsetData() : TopologySubsetDataImpl<Quad, VecT>()
    {}

    /// Constructor
    QuadSubsetData(typename TopologySubsetDataImpl<Quad, VecT>::size_type n, const value_type& value): TopologySubsetDataImpl<Quad, VecT>(n,value) {}

    /// Constructor
    explicit QuadSubsetData(typename TopologySubsetDataImpl<Quad, VecT>::size_type n): TopologySubsetDataImpl<Quad, VecT>(n) {}
    /// Constructor
    QuadSubsetData(const container_type& x): TopologySubsetDataImpl<Quad, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    QuadSubsetData(InputIterator first, InputIterator last): TopologySubsetDataImpl<Quad, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    QuadSubsetData(typename QuadSubsetData<VecT>::const_iterator first, typename QuadSubsetData<VecT>::const_iterator last): TopologySubsetDataImpl<Quad, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

protected:
    void linkToElementDataArray() {this->linkToQuadDataArray();}

};



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class TetrahedronSubsetData : public TopologySubsetDataImpl<Tetrahedron, VecT>
{
public:
    typedef typename TopologySubsetDataImpl<Tetrahedron, VecT>::container_type container_type;
    typedef typename TopologySubsetDataImpl<Tetrahedron, VecT>::value_type value_type;

    TetrahedronSubsetData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologySubsetDataImpl<Tetrahedron, VecT>(data)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TetrahedronSubsetData() : TopologySubsetDataImpl<Tetrahedron, VecT>()
    {}

    /// Constructor
    TetrahedronSubsetData(typename TopologySubsetDataImpl<Tetrahedron, VecT>::size_type n, const value_type& value): TopologySubsetDataImpl<Tetrahedron, VecT>(n,value) {}

    /// Constructor
    explicit TetrahedronSubsetData(typename TopologySubsetDataImpl<Tetrahedron, VecT>::size_type n): TopologySubsetDataImpl<Tetrahedron, VecT>(n) {}
    /// Constructor
    TetrahedronSubsetData(const container_type& x): TopologySubsetDataImpl<Tetrahedron, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TetrahedronSubsetData(InputIterator first, InputIterator last): TopologySubsetDataImpl<Tetrahedron, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TetrahedronSubsetData(typename TetrahedronSubsetData<VecT>::const_iterator first, typename TetrahedronSubsetData<VecT>::const_iterator last): TopologySubsetDataImpl<Tetrahedron, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

protected:
    void linkToElementDataArray() {this->linkToTetrahedronDataArray();}

};




////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Hexahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT >
class HexahedronSubsetData : public TopologySubsetDataImpl<Hexahedron, VecT>
{
public:
    typedef typename TopologySubsetDataImpl<Hexahedron, VecT>::container_type container_type;
    typedef typename TopologySubsetDataImpl<Hexahedron, VecT>::value_type value_type;

    HexahedronSubsetData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
        : TopologySubsetDataImpl<Hexahedron, VecT>(data)
    {}

    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    HexahedronSubsetData() : TopologySubsetDataImpl<Hexahedron, VecT>()
    {}

    /// Constructor
    HexahedronSubsetData(typename TopologySubsetDataImpl<Hexahedron, VecT>::size_type n, const value_type& value): TopologySubsetDataImpl<Hexahedron, VecT>(n,value) {}

    /// Constructor
    explicit HexahedronSubsetData(typename TopologySubsetDataImpl<Hexahedron, VecT>::size_type n): TopologySubsetDataImpl<Hexahedron, VecT>(n) {}
    /// Constructor
    HexahedronSubsetData(const container_type& x): TopologySubsetDataImpl<Hexahedron, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    HexahedronSubsetData(InputIterator first, InputIterator last): TopologySubsetDataImpl<Hexahedron, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    HexahedronSubsetData(typename HexahedronSubsetData<VecT>::const_iterator first, typename HexahedronSubsetData<VecT>::const_iterator last): TopologySubsetDataImpl<Hexahedron, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */

protected:
    void linkToElementDataArray() {this->linkToHexahedronDataArray();}

};





} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSUBSETDATA_H
