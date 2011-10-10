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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL

#include <sofa/component/topology/TopologyData.h>

#include <sofa/component/topology/EdgeSetTopologyChange.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>
#include <sofa/component/topology/QuadSetTopologyChange.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>

#include <sofa/component/topology/EdgeSetTopologyEngine.inl>

namespace sofa
{

namespace component
{

namespace topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////implementation//////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::registerTopologicalData()
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->registerTopology();
#ifndef NDEBUG // too much warnings
    else
        std::cout<<"Error: TopologyData: " << this->getName() << " has no engine. Use createTopologicalEngine function before." << std::endl;
#endif
#endif
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::addInputData(sofa::core::objectmodel::BaseData *_data)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (this->m_topologicalEngine)
        this->m_topologicalEngine->addInput(_data);
#ifndef NDEBUG // too much warnings
    else
        std::cout<<"Error: TopologyData: " << this->getName() << " has no engine. Use createTopologicalEngine function before." << std::endl;
#endif
#else
    (void)_data;
#endif
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::handleTopologyEvents( std::list< const core::topology::TopologyChange *>::const_iterator changeIt,
        std::list< const core::topology::TopologyChange *>::const_iterator &end )
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (this->m_topologicalEngine)
    {
        (void)changeIt;
        (void)end;
        return;
    }
#endif
    while( changeIt != end )
    {
        core::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();

        switch( changeType )
        {
        case core::topology::TETRAHEDRAADDED:
        {
            if (m_createTetrahedronFunc)
            {
                const TetrahedraAdded *ea=static_cast< const TetrahedraAdded* >( *changeIt );
                this->applyCreateTetrahedronFunction(ea->tetrahedronIndexArray);
            }
            break;
        }
        case core::topology::TETRAHEDRAREMOVED:
        {
            if (m_destroyTetrahedronFunc)
            {
                const TetrahedraRemoved *er=static_cast< const TetrahedraRemoved * >( *changeIt );
                this->applyDestroyTetrahedronFunction(er->getArray());
            }
            break;
        }
        case core::topology::TRIANGLESADDED:
        {
            if (m_createTriangleFunc)
            {
                const TrianglesAdded *ea=static_cast< const TrianglesAdded* >( *changeIt );
                this->applyCreateTriangleFunction(ea->triangleIndexArray);
            }
            break;
        }
        case core::topology::TRIANGLESREMOVED:
        {
            if (m_destroyTriangleFunc)
            {
                const TrianglesRemoved *er=static_cast< const TrianglesRemoved * >( *changeIt );
                this->applyDestroyTriangleFunction(er->getArray());
            }
            break;
        }
        case core::topology::QUADSADDED:
        {
            if (m_createQuadFunc)
            {
                const QuadsAdded *ea=static_cast< const QuadsAdded* >( *changeIt );
                this->applyCreateQuadFunction(ea->quadIndexArray);
            }
            break;
        }
        case core::topology::QUADSREMOVED:
        {
            if (m_destroyQuadFunc)
            {
                const QuadsRemoved *er=static_cast< const QuadsRemoved * >( *changeIt );
                this->applyDestroyQuadFunction(er->getArray());
            }
            break;
        }
        case core::topology::HEXAHEDRAADDED:
        {
            if (m_createHexahedronFunc)
            {
                const HexahedraAdded *ea=static_cast< const HexahedraAdded* >( *changeIt );
                this->applyCreateHexahedronFunction(ea->hexahedronIndexArray);
            }
            break;
        }
        case core::topology::HEXAHEDRAREMOVED:
        {
            if (m_destroyHexahedronFunc)
            {
                const HexahedraRemoved *er=static_cast< const HexahedraRemoved * >( *changeIt );
                this->applyDestroyHexahedronFunction(er->getArray());
            }
            break;
        }
        case core::topology::EDGESADDED:
        {
            const EdgesAdded *ea=static_cast< const EdgesAdded * >( *changeIt );
            add( ea->getNbAddedEdges(), ea->edgeArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::EDGESREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *changeIt ) )->getArray();
            remove( tab );
            break;
        }
        case core::topology::EDGESMOVED_REMOVING:
        {
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const EdgesMoved_Removing *>( *changeIt ) )->edgesAroundVertexMoved;
            container_type& data = *(this->beginEdit());

            for (unsigned int i = 0; i <edgeList.size(); i++)
                m_destroyFunc( edgeList[i], m_destroyParam, data[edgeList[i]] );

            this->endEdit();
            break;
        }
        case core::topology::EDGESMOVED_ADDING:
        {
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const EdgesMoved_Adding *>( *changeIt ) )->edgesAroundVertexMoved;
            const sofa::helper::vector< TopologyElementType >& edgeArray = ( static_cast< const EdgesMoved_Adding *>( *changeIt ) )->edgeArray2Moved;
            container_type& data = *(this->beginEdit());

            // Recompute data
            sofa::helper::vector< unsigned int > ancestors;
            sofa::helper::vector< double >  coefs;
            coefs.push_back (1.0);
            ancestors.resize(1);

            for (unsigned int i = 0; i <edgeList.size(); i++)
            {
                ancestors[0] = edgeList[i];
                m_createFunc( edgeList[i], m_createParam, data[edgeList[i]], edgeArray[i], ancestors, coefs );
            }

            this->endEdit();
            break;
        }
        case core::topology::TRIANGLESMOVED_REMOVING:
        {
            if (m_destroyTriangleFunc)
            {
                const TrianglesMoved_Removing *tm=static_cast< const TrianglesMoved_Removing* >( *changeIt );
                (*m_destroyTriangleFunc)(tm->trianglesAroundVertexMoved,m_createParam,*(this->beginEdit() ) );
                this->endEdit();
            }

            break;
        }
        case core::topology::TRIANGLESMOVED_ADDING:
        {
            if (m_createTriangleFunc)
            {
                const TrianglesMoved_Adding *tm=static_cast< const TrianglesMoved_Adding * >( *changeIt );
                (*m_createTriangleFunc)(tm->trianglesAroundVertexMoved,m_createParam,*(this->beginEdit() ) );
                this->endEdit();
            }

            break;
        }
        default:
            // Ignore events that are not Edge or Point related.
            break;
        }; // switch( changeType )

        ++changeIt;
    }
}



///////////////////// Public functions to call pointer to fonction ////////////////////////
/// Apply adding points elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyCreatePointFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_createPointFunc)
    {
        (*m_createPointFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}
/// Apply removing points elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyDestroyPointFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_destroyPointFunc)
    {
        (*m_destroyPointFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}

/// Apply adding edges elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyCreateEdgeFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_createEdgeFunc)
    {
        (*m_createEdgeFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}
/// Apply removing edges elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyDestroyEdgeFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_destroyEdgeFunc)
    {
        (*m_destroyEdgeFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}

/// Apply adding triangles elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyCreateTriangleFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_createTriangleFunc)
    {
        (*m_createTriangleFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}
/// Apply removing triangles elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyDestroyTriangleFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_destroyTriangleFunc)
    {
        (*m_destroyTriangleFunc)(indices,m_createParam, *(this->beginEdit() ) );
        this->endEdit();
    }
}

/// Apply adding quads elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyCreateQuadFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_createQuadFunc)
    {
        (*m_createQuadFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}
/// Apply removing quads elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyDestroyQuadFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_destroyQuadFunc)
    {
        (*m_destroyQuadFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}

/// Apply adding tetrahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyCreateTetrahedronFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_createTetrahedronFunc)
    {
        (*m_createTetrahedronFunc)(indices,m_createParam, *(this->beginEdit() ) );
        this->endEdit();
    }
}
/// Apply removing tetrahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyDestroyTetrahedronFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_destroyTetrahedronFunc)
    {
        (*m_destroyTetrahedronFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}

/// Apply adding hexahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyCreateHexahedronFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_createHexahedronFunc)
    {
        (*m_createHexahedronFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}
/// Apply removing hexahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyDestroyHexahedronFunction(const sofa::helper::vector<unsigned int> & indices)
{
    if (m_destroyHexahedronFunc)
    {
        (*m_destroyHexahedronFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}



/// Creation function, called when adding elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreateFunction(t_createFunc createFunc)
{
    m_createFunc=createFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToEdgeDataArray();
}
/// Destruction function, called when deleting elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyFunction(t_destroyFunc destroyFunc)
{
    m_destroyFunc=destroyFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToEdgeDataArray();
}

/// Creation function, called when adding points elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreatePointFunction(t_createPointFunc createPointFunc)
{
    m_createPointFunc=createPointFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToPointDataArray();
}

/// Destruction function, called when removing points elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyPointFunction(t_destroyPointFunc destroyPointFunc)
{
    m_destroyPointFunc=destroyPointFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToPointDataArray();
}

/// Creation function, called when adding edges elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreateEdgeFunction(t_createEdgeFunc createEdgeFunc)
{
    m_createEdgeFunc=createEdgeFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToEdgeDataArray();
}

/// Destruction function, called when removing edges elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyEdgeFunction(t_destroyEdgeFunc destroyEdgeFunc)
{
    m_destroyEdgeFunc=destroyEdgeFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToEdgeDataArray();
}

/// Creation function, called when adding triangles elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreateTriangleFunction(t_createTriangleFunc createTriangleFunc)
{
    m_createTriangleFunc=createTriangleFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToTriangleDataArray();
}
/// Destruction function, called when removing triangles elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyTriangleFunction(t_destroyTriangleFunc destroyTriangleFunc)
{
    m_destroyTriangleFunc=destroyTriangleFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToTriangleDataArray();
}

/// Creation function, called when adding quads elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreateQuadFunction(t_createQuadFunc createQuadFunc)
{
    m_createQuadFunc=createQuadFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToQuadDataArray();
}
/// Destruction function, called when removing quads elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyQuadFunction(t_destroyQuadFunc destroyQuadFunc)
{
    m_destroyQuadFunc=destroyQuadFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToQuadDataArray();
}

/// Creation function, called when adding tetrahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreateTetrahedronFunction(t_createTetrahedronFunc createTetrahedronFunc)
{
    m_createTetrahedronFunc=createTetrahedronFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToTetrahedronDataArray();
}
/// Destruction function, called when removing tetrahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyTetrahedronFunction(t_destroyTetrahedronFunc destroyTetrahedronFunc)
{
    m_destroyTetrahedronFunc=destroyTetrahedronFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToTetrahedronDataArray();
}

/// Creation function, called when adding hexahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreateHexahedronFunction(t_createHexahedronFunc createHexahedronFunc)
{
    m_createHexahedronFunc=createHexahedronFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToHexahedronDataArray();
}
/// Destruction function, called when removing hexahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyHexahedronFunction(t_destroyHexahedronFunc destroyHexahedronFunc)
{
    m_destroyHexahedronFunc=destroyHexahedronFunc;

    if(m_topologicalEngine)
        m_topologicalEngine->linkToHexahedronDataArray();
}

/// Creation function, called when adding parameter to those elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyParameter( void* destroyParam )
{
    m_destroyParam=destroyParam;
}
/// Destruction function, called when removing parameter to those elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setCreateParameter( void* createParam )
{
    m_createParam=createParam;
}




///////////////////// Private functions on TopologyData changes /////////////////////////////

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::swap( unsigned int i1, unsigned int i2 )
{
    container_type& data = *(this->beginEdit());
    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;
    this->endEdit();
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::add(unsigned int nbElements,
        const sofa::helper::vector<TopologyElementType> &elem,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    // Using default values
    container_type& data = *(this->beginEdit());
    unsigned int i0 = data.size();
    data.resize(i0+nbElements);

    for (unsigned int i = 0; i < nbElements; ++i)
    {
        value_type& t = data[i0+i];
        if (ancestors.empty() || coefs.empty())
        {
            const sofa::helper::vector< unsigned int > empty_vecint;
            const sofa::helper::vector< double > empty_vecdouble;
            m_createFunc( i0+i, m_createParam, t, elem[i], empty_vecint, empty_vecdouble);
        }
        else
            m_createFunc( i0+i, m_createParam, t, elem[i], ancestors[i], coefs[i] );
    }
    this->endEdit();
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::remove( const sofa::helper::vector<unsigned int> &index )
{

    container_type& data = *(this->beginEdit());
    unsigned int last = data.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        m_destroyFunc( index[i], m_destroyParam, data[index[i]] );
        swap( index[i], last );
        --last;
    }

    data.resize( data.size() - index.size() );
    this->endEdit();
    //sout << "EdgeData: vector has now "<<this->size()<<" entries."<<sendl;
}





template< class VecT >
void PointDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new PointSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< class VecT >
void EdgeDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new EdgeSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< class VecT >
void TriangleDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new TriangleSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< class VecT >
void QuadDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new QuadSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< class VecT >
void TetrahedronDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new TetrahedronSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< class VecT >
void HexahedronDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new HexahedronSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
