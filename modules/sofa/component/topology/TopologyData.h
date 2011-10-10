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


/** \brief Basic creation function for element of type T : simply calls default constructor.
*
*/
template < typename TopologyElementType, typename VecT >
inline void topo_basicCreateFunc(unsigned int ,
        void* , VecT& t,
        const TopologyElementType& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    t = VecT();
    //return;
}

/** \brief Basic destruction function for element of type T : does nothing.
*
*/
template < typename TopologyElementType, typename VecT >
inline void topo_basicDestroyFunc(unsigned int , void* , VecT& )
{
    return;
}


/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/
template< class TopologyElementType, class VecT>
class TopologyData : public sofa::core::topology::BaseTopologyData<VecT>
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



    /// Creation function, called when adding elements.
    typedef void (*t_createFunc)(unsigned int, void*, value_type&, const TopologyElementType&, const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >& );
    /// Destruction function, called when deleting elements.
    typedef void (*t_destroyFunc)(unsigned int, void*, value_type&);

    /// Creation function, called when adding points elements.
    typedef void (*t_createPointFunc)(const sofa::helper::vector<unsigned int> &, void*,  container_type &);
    /// Destruction function, called when removing points elements.
    typedef void (*t_destroyPointFunc)(const sofa::helper::vector<unsigned int> &, void*,  container_type &);

    /// Creation function, called when adding edges elements.
    typedef void (*t_createEdgeFunc)(const sofa::helper::vector<unsigned int> &, void*,  container_type &);
    /// Destruction function, called when removing edges elements.
    typedef void (*t_destroyEdgeFunc)(const sofa::helper::vector<unsigned int> &, void*,  container_type &);

    /// Creation function, called when adding triangles elements.
    typedef void (*t_createTriangleFunc)(const sofa::helper::vector<unsigned int> &, void*,  container_type &);
    /// Destruction function, called when removing triangles elements.
    typedef void (*t_destroyTriangleFunc)(const sofa::helper::vector<unsigned int> &, void*,  container_type &);

    /// Creation function, called when adding quads elements.
    typedef void (*t_createQuadFunc)(const sofa::helper::vector<unsigned int> &, void*, container_type &);
    /// Destruction function, called when removing quads elements.
    typedef void (*t_destroyQuadFunc)(const sofa::helper::vector<unsigned int> &, void*, container_type &);

    /// Creation function, called when adding tetrahedra elements.
    typedef void (*t_createTetrahedronFunc)(const sofa::helper::vector<unsigned int> &, void*, container_type &);
    /// Destruction function, called when removing tetrahedra elements.
    typedef void (*t_destroyTetrahedronFunc)(const sofa::helper::vector<unsigned int> &, void*, container_type &);

    /// Creation function, called when adding hexahedra elements.
    typedef void (*t_createHexahedronFunc)(const sofa::helper::vector<unsigned int> &, void*, container_type &);
    /// Destruction function, called when removing hexahedra elements.
    typedef void (*t_destroyHexahedronFunc)(const sofa::helper::vector<unsigned int> &, void*, container_type &);

    /// Constructors
public:
    /// Constructor
    TopologyData( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data,
            void (*createFunc) (unsigned int, void*, value_type&, const TopologyElementType &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topo_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topo_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : sofa::core::topology::BaseTopologyData< VecT >(data),
          m_createFunc(createFunc), m_destroyFunc(destroyFunc),
          m_createPointFunc(0), m_destroyPointFunc(0),
          m_createEdgeFunc(0), m_destroyEdgeFunc(0),
          m_createTriangleFunc(0), m_destroyTriangleFunc(0),
          m_createQuadFunc(0), m_destroyQuadFunc(0),
          m_createTetrahedronFunc(0), m_destroyTetrahedronFunc(0),
          m_createHexahedronFunc(0), m_destroyHexahedronFunc(0),
          m_createParam(createParam), m_destroyParam(destroyParam),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    TopologyData(size_type n, const value_type& value) : sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n, value);
        this->endEdit();
    }
    /// Constructor
    TopologyData(int n, const value_type& value) : sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n, value);
        this->endEdit();
    }
    /// Constructor
    TopologyData(long n, const value_type& value): sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n, value);
        this->endEdit();
    }
    /// Constructor
    explicit TopologyData(size_type n): sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        data->resize(n);
        this->endEdit();
    }
    /// Constructor
    TopologyData(const container_type& x): sofa::core::topology::BaseTopologyData< container_type >(0, false, false), m_topologicalEngine(NULL)
    {
        container_type* data = this->beginEdit();
        (*data) = x;
        this->endEdit();
    }

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TopologyData(InputIterator first, InputIterator last): sofa::core::topology::BaseTopologyData< container_type >(0, false, false)
    {
        container_type* data = this->beginEdit();
        data->assign(first, last);
        this->endEdit();
    }
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    TopologyData(const_iterator first, const_iterator last): sofa::core::topology::BaseTopologyData< container_type >(0, false, false)
    {
        container_type* data = this->beginEdit();
        data->assign(first, last);
        this->endEdit();
    }
#endif /* __STL_MEMBER_TEMPLATES */


    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TopologyData( void (*createFunc) (unsigned int, void*, value_type&, const TopologyElementType &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topo_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topo_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : sofa::core::topology::BaseTopologyData< VecT >(0, false, false),
          m_createFunc(createFunc), m_destroyFunc(destroyFunc),
          m_createPointFunc(0), m_destroyPointFunc(0),
          m_createEdgeFunc(0), m_destroyEdgeFunc(0),
          m_createTriangleFunc(0), m_destroyTriangleFunc(0),
          m_createQuadFunc(0), m_destroyQuadFunc(0),
          m_createTetrahedronFunc(0), m_destroyTetrahedronFunc(0),
          m_createHexahedronFunc(0), m_destroyHexahedronFunc(0),
          m_createParam(createParam), m_destroyParam(destroyParam),
          m_topologicalEngine(NULL)
    {}


    /// Handle EdgeSetTopology related events, ignore others. DEPRECATED
    void handleTopologyEvents( std::list< const core::topology::TopologyChange *>::const_iterator changeIt,
            std::list< const core::topology::TopologyChange *>::const_iterator &end );



    /// Creation function, called when adding elements.
    void setCreateFunction(t_createFunc createFunc);
    /// Destruction function, called when deleting elements.
    void setDestroyFunction(t_destroyFunc destroyFunc);

    /// Creation function, called when adding parameter to those elements.
    void setDestroyParameter( void* destroyParam );
    /// Destruction function, called when removing parameter to those elements.
    void setCreateParameter( void* createParam );


    /// Creation function, called when adding points elements.
    void setCreatePointFunction(t_createPointFunc createPointFunc);
    /// Destruction function, called when removing points elements.
    void setDestroyPointFunction(t_destroyPointFunc destroyPointFunc);

    /// Creation function, called when adding edges elements.
    void setCreateEdgeFunction(t_createEdgeFunc createEdgeFunc);
    /// Destruction function, called when removing edges elements.
    void setDestroyEdgeFunction(t_destroyEdgeFunc destroyEdgeFunc);

    /// Creation function, called when adding triangles elements.
    void setCreateTriangleFunction(t_createTriangleFunc createTriangleFunc);
    /// Destruction function, called when removing triangles elements.
    void setDestroyTriangleFunction(t_destroyTriangleFunc destroyTriangleFunc);

    /// Creation function, called when adding quads elements.
    void setCreateQuadFunction(t_createQuadFunc createQuadFunc);
    /// Destruction function, called when removing quads elements.
    void setDestroyQuadFunction(t_destroyQuadFunc destroyQuadFunc);

    /// Creation function, called when adding tetrahedra elements.
    void setCreateTetrahedronFunction(t_createTetrahedronFunc createTetrahedronFunc);
    /// Destruction function, called when removing tetrahedra elements.
    void setDestroyTetrahedronFunction(t_destroyTetrahedronFunc destroyTetrahedronFunc);

    /// Creation function, called when adding hexahedra elements.
    void setCreateHexahedronFunction(t_createHexahedronFunc createHexahedronFunc);
    /// Destruction function, called when removing hexahedra elements.
    void setDestroyHexahedronFunction(t_destroyHexahedronFunc destroyHexahedronFunc);




    /** Public fonction to apply creation and destruction functions */
    /// Apply adding points elements.
    void applyCreatePointFunction(unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elem,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing points elements.
    void applyDestroyPointFunction(const sofa::helper::vector<unsigned int> & indices);

    /// Apply adding edges elements.
    void applyCreateEdgeFunction(unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elem,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing edges elements.
    void applyDestroyEdgeFunction(const sofa::helper::vector<unsigned int> & indices);

    /// Apply adding triangles elements.
    void applyCreateTriangleFunction(unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elem,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing triangles elements.
    void applyDestroyTriangleFunction(const sofa::helper::vector<unsigned int> & indices);

    /// Apply adding quads elements.
    void applyCreateQuadFunction(unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elem,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing quads elements.
    void applyDestroyQuadFunction(const sofa::helper::vector<unsigned int> & indices);

    /// Apply adding tetrahedra elements.
    void applyCreateTetrahedronFunction(unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elem,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing tetrahedra elements.
    void applyDestroyTetrahedronFunction(const sofa::helper::vector<unsigned int> & indices);

    /// Apply adding hexahedra elements.
    void applyCreateHexahedronFunction(unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elem,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);
    /// Apply removing hexahedra elements.
    void applyDestroyHexahedronFunction(const sofa::helper::vector<unsigned int> & indices);



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

protected:
    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int i1, unsigned int i2 );

    /// Add some values. Values are added at the end of the vector.
    virtual void add( unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& elem,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    /// Remove the values corresponding to the Edges removed.
    virtual void remove( const sofa::helper::vector<unsigned int> &index );


    t_createFunc m_createFunc;
    t_destroyFunc m_destroyFunc;
    t_createPointFunc m_createPointFunc;
    t_destroyPointFunc m_destroyPointFunc;
    t_createEdgeFunc m_createEdgeFunc;
    t_destroyEdgeFunc m_destroyEdgeFunc;
    t_createTriangleFunc m_createTriangleFunc;
    t_destroyTriangleFunc m_destroyTriangleFunc;
    t_createQuadFunc m_createQuadFunc;
    t_destroyQuadFunc m_destroyQuadFunc;
    t_createTetrahedronFunc m_createTetrahedronFunc;
    t_destroyTetrahedronFunc m_destroyTetrahedronFunc;
    t_createHexahedronFunc m_createHexahedronFunc;
    t_destroyHexahedronFunc m_destroyHexahedronFunc;


    /** Parameter to be passed to creation function.
    *
    * Warning : construction and destruction of this object is not of the responsibility of EdgeData.
    */
    void* m_createParam;

    /** Parameter to be passed to destruction function.
    *
    * Warning : construction and destruction of this object is not of the responsibility of EdgeData.
    */
    void* m_destroyParam;

    sofa::core::topology::TopologyEngine* m_topologicalEngine;
};




// Topology specialization
template < typename VecT >
inline void topoPoint_basicCreateFunc(unsigned int ,
        void* , VecT& t,
        const Point& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    t = VecT();
    //return;
}

template < typename VecT >
inline void topoPoint_basicDestroyFunc(unsigned int , void* , VecT& )
{
    return;
}


template< class VecT >
class PointDataNew : public TopologyData<Point, VecT>
{
public:
    typedef typename TopologyData<Point, VecT>::container_type container_type;
    typedef typename TopologyData<Point, VecT>::value_type value_type;

    PointDataNew( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data,
            void (*createFunc) (unsigned int, void*, value_type&, const Point &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoPoint_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoPoint_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Point, VecT>(data, createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    PointDataNew(typename TopologyData<Point, VecT>::size_type n, const value_type& value): TopologyData<Point, VecT>(n,value) {}
    /// Constructor
    PointDataNew(int n, const value_type& value): TopologyData<Point, VecT>(n,value) {}
    /// Constructor
    PointDataNew(long n, const value_type& value): TopologyData<Point, VecT>(n,value) {}
    /// Constructor
    explicit PointDataNew(typename TopologyData<Point, VecT>::size_type n): TopologyData<Point, VecT>(n) {}
    /// Constructor
    PointDataNew(const container_type& x): TopologyData<Point, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    PointDataNew(InputIterator first, InputIterator last): TopologyData<Point, VecT>(first,last) {}
#else /* __STL_MEMBER_TEMPLATES */
    /// Constructor
    PointDataNew(typename PointDataNew<VecT>::const_iterator first, typename PointDataNew<VecT>::const_iterator last): TopologyData<Point, VecT>(first,last) {}
#endif /* __STL_MEMBER_TEMPLATES */


    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    PointDataNew( void (*createFunc) (unsigned int, void*, value_type&, const Point &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoPoint_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoPoint_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Point, VecT>(createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    /** Public functions to handle topological engine creation */
    /// To create topological engine link to this Data. Pointer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

protected:
    PointSetTopologyEngine<VecT>* m_topologicalEngine;

};



template < typename VecT >
inline void topoEdge_basicCreateFunc(unsigned int ,
        void* , VecT& t,
        const Edge& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    t = VecT();
    //return;
}

template < typename VecT >
inline void topoEdge_basicDestroyFunc(unsigned int , void* , VecT& )
{
    return;
}


template< class VecT >
class EdgeDataNew : public TopologyData<Edge, VecT>
{
public:
    typedef typename TopologyData<Edge, VecT>::container_type container_type;
    typedef typename TopologyData<Edge, VecT>::value_type value_type;

    EdgeDataNew( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data,
            void (*createFunc) (unsigned int, void*, value_type&, const Edge &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoEdge_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoEdge_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Edge, VecT>(data, createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    EdgeDataNew(typename TopologyData<Edge, VecT>::size_type n, const value_type& value): TopologyData<Edge, VecT>(n,value) {}
    /// Constructor
    EdgeDataNew(int n, const value_type& value): TopologyData<Edge, VecT>(n,value) {}
    /// Constructor
    EdgeDataNew(long n, const value_type& value): TopologyData<Edge, VecT>(n,value) {}
    /// Constructor
    explicit EdgeDataNew(typename TopologyData<Edge, VecT>::size_type n): TopologyData<Edge, VecT>(n) {}
    /// Constructor
    EdgeDataNew(const container_type& x): TopologyData<Edge, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    EdgeDataNew(InputIterator first, InputIterator last): TopologyData<Edge, VecT>(first,last) {}
#else // __STL_MEMBER_TEMPLATES
    /// Constructor
    EdgeDataNew(typename EdgeDataNew<VecT>::const_iterator first, typename EdgeDataNew<VecT>::const_iterator last): TopologyData<Edge, VecT>(first,last) {}
#endif // __STL_MEMBER_TEMPLATES


    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    EdgeDataNew( void (*createFunc) (unsigned int, void*, value_type&, const Edge &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoEdge_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoEdge_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Edge, VecT>(createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    // Public functions to handle topological engine creation
    /// To create topological engine link to this Data. Edgeer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

protected:
    EdgeSetTopologyEngine<VecT>* m_topologicalEngine;

};



template < typename VecT >
inline void topoTriangle_basicCreateFunc(unsigned int ,
        void* , VecT& t,
        const Triangle& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    t = VecT();
    //return;
}

template < typename VecT >
inline void topoTriangle_basicDestroyFunc(unsigned int , void* , VecT& )
{
    return;
}


template< class VecT >
class TriangleDataNew : public TopologyData<Triangle, VecT>
{
public:
    typedef typename TopologyData<Triangle, VecT>::container_type container_type;
    typedef typename TopologyData<Triangle, VecT>::value_type value_type;

    TriangleDataNew( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data,
            void (*createFunc) (unsigned int, void*, value_type&, const Triangle &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoTriangle_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoTriangle_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Triangle, VecT>(data, createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    TriangleDataNew(typename TopologyData<Triangle, VecT>::size_type n, const value_type& value): TopologyData<Triangle, VecT>(n,value) {}
    /// Constructor
    TriangleDataNew(int n, const value_type& value): TopologyData<Triangle, VecT>(n,value) {}
    /// Constructor
    TriangleDataNew(long n, const value_type& value): TopologyData<Triangle, VecT>(n,value) {}
    /// Constructor
    explicit TriangleDataNew(typename TopologyData<Triangle, VecT>::size_type n): TopologyData<Triangle, VecT>(n) {}
    /// Constructor
    TriangleDataNew(const container_type& x): TopologyData<Triangle, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TriangleDataNew(InputIterator first, InputIterator last): TopologyData<Triangle, VecT>(first,last) {}
#else // __STL_MEMBER_TEMPLATES
    /// Constructor
    TriangleDataNew(typename TriangleDataNew<VecT>::const_iterator first, typename TriangleDataNew<VecT>::const_iterator last): TopologyData<Triangle, VecT>(first,last) {}
#endif // __STL_MEMBER_TEMPLATES


    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TriangleDataNew( void (*createFunc) (unsigned int, void*, value_type&, const Triangle &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoTriangle_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoTriangle_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Triangle, VecT>(createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    // Public functions to handle topological engine creation
    /// To create topological engine link to this Data. Triangleer to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

protected:
    TriangleSetTopologyEngine<VecT>* m_topologicalEngine;

};



template < typename VecT >
inline void topoQuad_basicCreateFunc(unsigned int ,
        void* , VecT& t,
        const Quad& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    t = VecT();
    //return;
}

template < typename VecT >
inline void topoQuad_basicDestroyFunc(unsigned int , void* , VecT& )
{
    return;
}


template< class VecT >
class QuadDataNew : public TopologyData<Quad, VecT>
{
public:
    typedef typename TopologyData<Quad, VecT>::container_type container_type;
    typedef typename TopologyData<Quad, VecT>::value_type value_type;

    QuadDataNew( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data,
            void (*createFunc) (unsigned int, void*, value_type&, const Quad &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoQuad_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoQuad_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Quad, VecT>(data, createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    QuadDataNew(typename TopologyData<Quad, VecT>::size_type n, const value_type& value): TopologyData<Quad, VecT>(n,value) {}
    /// Constructor
    QuadDataNew(int n, const value_type& value): TopologyData<Quad, VecT>(n,value) {}
    /// Constructor
    QuadDataNew(long n, const value_type& value): TopologyData<Quad, VecT>(n,value) {}
    /// Constructor
    explicit QuadDataNew(typename TopologyData<Quad, VecT>::size_type n): TopologyData<Quad, VecT>(n) {}
    /// Constructor
    QuadDataNew(const container_type& x): TopologyData<Quad, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    QuadDataNew(InputIterator first, InputIterator last): TopologyData<Quad, VecT>(first,last) {}
#else // __STL_MEMBER_TEMPLATES
    /// Constructor
    QuadDataNew(typename QuadDataNew<VecT>::const_iterator first, typename QuadDataNew<VecT>::const_iterator last): TopologyData<Quad, VecT>(first,last) {}
#endif // __STL_MEMBER_TEMPLATES


    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    QuadDataNew( void (*createFunc) (unsigned int, void*, value_type&, const Quad &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoQuad_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoQuad_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Quad, VecT>(createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    // Public functions to handle topological engine creation
    /// To create topological engine link to this Data. Quader to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

protected:
    QuadSetTopologyEngine<VecT>* m_topologicalEngine;

};



template < typename VecT >
inline void topoTetrahedron_basicCreateFunc(unsigned int ,
        void* , VecT& t,
        const Tetrahedron& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    t = VecT();
    //return;
}

template < typename VecT >
inline void topoTetrahedron_basicDestroyFunc(unsigned int , void* , VecT& )
{
    return;
}


template< class VecT >
class TetrahedronDataNew : public TopologyData<Tetrahedron, VecT>
{
public:
    typedef typename TopologyData<Tetrahedron, VecT>::container_type container_type;
    typedef typename TopologyData<Tetrahedron, VecT>::value_type value_type;

    TetrahedronDataNew( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data,
            void (*createFunc) (unsigned int, void*, value_type&, const Tetrahedron &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoTetrahedron_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoTetrahedron_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Tetrahedron, VecT>(data, createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    TetrahedronDataNew(typename TopologyData<Tetrahedron, VecT>::size_type n, const value_type& value): TopologyData<Tetrahedron, VecT>(n,value) {}
    /// Constructor
    TetrahedronDataNew(int n, const value_type& value): TopologyData<Tetrahedron, VecT>(n,value) {}
    /// Constructor
    TetrahedronDataNew(long n, const value_type& value): TopologyData<Tetrahedron, VecT>(n,value) {}
    /// Constructor
    explicit TetrahedronDataNew(typename TopologyData<Tetrahedron, VecT>::size_type n): TopologyData<Tetrahedron, VecT>(n) {}
    /// Constructor
    TetrahedronDataNew(const container_type& x): TopologyData<Tetrahedron, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    TetrahedronDataNew(InputIterator first, InputIterator last): TopologyData<Tetrahedron, VecT>(first,last) {}
#else // __STL_MEMBER_TEMPLATES
    /// Constructor
    TetrahedronDataNew(typename TetrahedronDataNew<VecT>::const_iterator first, typename TetrahedronDataNew<VecT>::const_iterator last): TopologyData<Tetrahedron, VecT>(first,last) {}
#endif // __STL_MEMBER_TEMPLATES


    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    TetrahedronDataNew( void (*createFunc) (unsigned int, void*, value_type&, const Tetrahedron &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoTetrahedron_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoTetrahedron_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Tetrahedron, VecT>(createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    // Public functions to handle topological engine creation
    /// To create topological engine link to this Data. Tetrahedroner to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

protected:
    TetrahedronSetTopologyEngine<VecT>* m_topologicalEngine;

};



template < typename VecT >
inline void topoHexahedron_basicCreateFunc(unsigned int ,
        void* , VecT& t,
        const Hexahedron& ,
        const sofa::helper::vector< unsigned int > &,
        const sofa::helper::vector< double >&)
{
    t = VecT();
    //return;
}

template < typename VecT >
inline void topoHexahedron_basicDestroyFunc(unsigned int , void* , VecT& )
{
    return;
}


template< class VecT >
class HexahedronDataNew : public TopologyData<Hexahedron, VecT>
{
public:
    typedef typename TopologyData<Hexahedron, VecT>::container_type container_type;
    typedef typename TopologyData<Hexahedron, VecT>::value_type value_type;

    HexahedronDataNew( const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data,
            void (*createFunc) (unsigned int, void*, value_type&, const Hexahedron &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoHexahedron_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoHexahedron_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Hexahedron, VecT>(data, createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    /// Constructor
    HexahedronDataNew(typename TopologyData<Hexahedron, VecT>::size_type n, const value_type& value): TopologyData<Hexahedron, VecT>(n,value) {}
    /// Constructor
    HexahedronDataNew(int n, const value_type& value): TopologyData<Hexahedron, VecT>(n,value) {}
    /// Constructor
    HexahedronDataNew(long n, const value_type& value): TopologyData<Hexahedron, VecT>(n,value) {}
    /// Constructor
    explicit HexahedronDataNew(typename TopologyData<Hexahedron, VecT>::size_type n): TopologyData<Hexahedron, VecT>(n) {}
    /// Constructor
    HexahedronDataNew(const container_type& x): TopologyData<Hexahedron, VecT>(x) {}

#ifdef __STL_MEMBER_TEMPLATES
    /// Constructor
    template <class InputIterator>
    HexahedronDataNew(InputIterator first, InputIterator last): TopologyData<Hexahedron, VecT>(first,last) {}
#else // __STL_MEMBER_TEMPLATES
    /// Constructor
    HexahedronDataNew(typename HexahedronDataNew<VecT>::const_iterator first, typename HexahedronDataNew<VecT>::const_iterator last): TopologyData<Hexahedron, VecT>(first,last) {}
#endif // __STL_MEMBER_TEMPLATES


    /// Optionnaly takes 2 parameters, a creation and a destruction function that will be called when adding/deleting elements.
    HexahedronDataNew( void (*createFunc) (unsigned int, void*, value_type&, const Hexahedron &,const sofa::helper::vector< unsigned int >&, const sofa::helper::vector< double >&) = topoHexahedron_basicCreateFunc,
            void* createParam  = (void*)NULL,
            void (*destroyFunc)(unsigned int, void*, value_type&) = topoHexahedron_basicDestroyFunc,
            void* destroyParam = (void*)NULL )
        : TopologyData<Hexahedron, VecT>(createFunc, createParam, destroyFunc, destroyParam),
          m_topologicalEngine(NULL)
    {}

    // Public functions to handle topological engine creation
    /// To create topological engine link to this Data. Hexahedroner to current topology is needed.
    void createTopologicalEngine(sofa::core::topology::BaseMeshTopology* _topology);

protected:
    HexahedronSetTopologyEngine<VecT>* m_topologicalEngine;

};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_H
