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


// WARNING: This function is deprecated
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
            ///////////////////////// Events on Points //////////////////////////////////////
        case core::topology::POINTSINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const PointsIndicesSwap * >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];

            this->swap( i1, i2 );
            break;
        }
        case core::topology::POINTSADDED:
        {
            const
            unsigned int nbPoints = ( static_cast< const PointsAdded * >( *changeIt ) )->getNbAddedVertices();
            sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors = ( static_cast< const PointsAdded * >( *changeIt ) )->ancestorsList;
            sofa::helper::vector< sofa::helper::vector< double       > > coefs     = ( static_cast< const PointsAdded * >( *changeIt ) )->coefs;

            this->applyCreatePointFunction(nbPoints, ancestors, coefs);
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();

            this->applyDestroyPointFunction( tab );
            break;
        }
        case core::topology::POINTSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getIndexArray();

            this->renumber( tab );
            break;
        }
        case core::topology::POINTSMOVED:
        {
            const sofa::helper::vector< unsigned int >& indexList = ( static_cast< const PointsMoved * >( *changeIt ) )->indicesList;
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors = ( static_cast< const PointsMoved * >( *changeIt ) )->ancestorsList;
            sofa::helper::vector< sofa::helper::vector< double > > coefs = ( static_cast< const PointsMoved * >( *changeIt ) )->baryCoefsList;

            this->move( indexList, ancestors, coefs);
            break;
        }

        ///////////////////////// Events on Edges //////////////////////////////////////
        case core::topology::EDGESINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const EdgesIndicesSwap * >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const EdgesIndicesSwap* >( *changeIt ) )->index[1];

            this->swap( i1, i2 );
            break;
        }
        case core::topology::EDGESADDED:
        {
            const EdgesAdded *ea=static_cast< const EdgesAdded * >( *changeIt );

            this->applyCreateEdgeFunction( ea->getNbAddedEdges(), ea->edgeArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::EDGESREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *changeIt ) )->getArray();

            this->applyDestroyEdgeFunction( tab );
            break;
        }
        case core::topology::EDGESMOVED_REMOVING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const EdgesMoved_Removing *>( *changeIt ) )->edgesAroundVertexMoved;
            container_type& data = *(this->beginEdit());

            for (unsigned int i = 0; i <edgeList.size(); i++)
                m_destroyFunc( edgeList[i], m_destroyParam, data[edgeList[i]] );

            this->endEdit();
            */
            break;
        }
        case core::topology::EDGESMOVED_ADDING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const EdgesMoved_Adding *>( *changeIt ) )->edgesAroundVertexMoved;
            const sofa::helper::vector< Edge >& edgeArray = ( static_cast< const EdgesMoved_Adding *>( *changeIt ) )->edgeArray2Moved;
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
            */
            break;
        }
        case core::topology::EDGESRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const EdgesRenumbering * >( *changeIt ) )->getIndexArray();

            this->renumber( tab );
            break;
        }

        ///////////////////////// Events on Triangles //////////////////////////////////////
        case core::topology::TRIANGLESINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const TrianglesIndicesSwap * >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const TrianglesIndicesSwap* >( *changeIt ) )->index[1];

            this->swap( i1, i2 );
            break;
        }
        case core::topology::TRIANGLESADDED:
        {
            const TrianglesAdded *ea=static_cast< const TrianglesAdded * >( *changeIt );

            this->applyCreateTriangleFunction( ea->getNbAddedTriangles(), ea->triangleArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::TRIANGLESREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TrianglesRemoved *>( *changeIt ) )->getArray();

            this->applyDestroyTriangleFunction( tab );
            break;
        }
        case core::topology::TRIANGLESMOVED_REMOVING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const TrianglesMoved_Removing *>( *changeIt ) )->edgesAroundVertexMoved;
            container_type& data = *(this->beginEdit());

            for (unsigned int i = 0; i <edgeList.size(); i++)
                m_destroyFunc( edgeList[i], m_destroyParam, data[edgeList[i]] );

            this->endEdit();
            */
            break;
        }
        case core::topology::TRIANGLESMOVED_ADDING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const TrianglesMoved_Adding *>( *changeIt ) )->edgesAroundVertexMoved;
            const sofa::helper::vector< Triangle >& edgeArray = ( static_cast< const TrianglesMoved_Adding *>( *changeIt ) )->edgeArray2Moved;
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
            */
            break;
        }
        case core::topology::TRIANGLESRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const TrianglesRenumbering * >( *changeIt ) )->getIndexArray();

            this->renumber( tab );
            break;
        }

        ///////////////////////// Events on Quads //////////////////////////////////////
        case core::topology::QUADSINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const QuadsIndicesSwap * >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const QuadsIndicesSwap* >( *changeIt ) )->index[1];

            this->swap( i1, i2 );
            break;
        }
        case core::topology::QUADSADDED:
        {
            const QuadsAdded *ea=static_cast< const QuadsAdded * >( *changeIt );

            this->applyCreateQuadFunction( ea->getNbAddedQuads(), ea->quadArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::QUADSREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const QuadsRemoved *>( *changeIt ) )->getArray();

            this->applyDestroyQuadFunction( tab );
            break;
        }
        case core::topology::QUADSMOVED_REMOVING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const QuadsMoved_Removing *>( *changeIt ) )->edgesAroundVertexMoved;
            container_type& data = *(this->beginEdit());

            for (unsigned int i = 0; i <edgeList.size(); i++)
                m_destroyFunc( edgeList[i], m_destroyParam, data[edgeList[i]] );

            this->endEdit();
            */
            break;
        }
        case core::topology::QUADSMOVED_ADDING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const QuadsMoved_Adding *>( *changeIt ) )->edgesAroundVertexMoved;
            const sofa::helper::vector< Quad >& edgeArray = ( static_cast< const QuadsMoved_Adding *>( *changeIt ) )->edgeArray2Moved;
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
            */
            break;
        }
        case core::topology::QUADSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const QuadsRenumbering * >( *changeIt ) )->getIndexArray();

            this->renumber( tab );
            break;
        }

        ///////////////////////// Events on Tetrahedra //////////////////////////////////////
        case core::topology::TETRAHEDRAINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const TetrahedraIndicesSwap * >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const TetrahedraIndicesSwap* >( *changeIt ) )->index[1];

            this->swap( i1, i2 );
            break;
        }
        case core::topology::TETRAHEDRAADDED:
        {
            const TetrahedraAdded *ea=static_cast< const TetrahedraAdded * >( *changeIt );

            this->applyCreateTetrahedronFunction( ea->getNbAddedTetrahedra(), ea->tetrahedronArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::TETRAHEDRAREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *changeIt ) )->getArray();

            this->applyDestroyTetrahedronFunction( tab );
            break;
        }
        case core::topology::TETRAHEDRAMOVED_REMOVING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const TetrahedraMoved_Removing *>( *changeIt ) )->edgesAroundVertexMoved;
            container_type& data = *(this->beginEdit());

            for (unsigned int i = 0; i <edgeList.size(); i++)
                m_destroyFunc( edgeList[i], m_destroyParam, data[edgeList[i]] );

            this->endEdit();
            */
            break;
        }
        case core::topology::TETRAHEDRAMOVED_ADDING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const TetrahedraMoved_Adding *>( *changeIt ) )->edgesAroundVertexMoved;
            const sofa::helper::vector< Edge >& edgeArray = ( static_cast< const TetrahedraMoved_Adding *>( *changeIt ) )->edgeArray2Moved;
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
            */
            break;
        }
        case core::topology::TETRAHEDRARENUMBERING:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const TetrahedraRenumbering * >( *changeIt ) )->getIndexArray();

            this->renumber( tab );
            break;
        }

        ///////////////////////// Events on Hexahedra //////////////////////////////////////
        case core::topology::HEXAHEDRAINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const HexahedraIndicesSwap * >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const HexahedraIndicesSwap* >( *changeIt ) )->index[1];

            this->swap( i1, i2 );
            break;
        }
        case core::topology::HEXAHEDRAADDED:
        {
            const HexahedraAdded *ea=static_cast< const HexahedraAdded * >( *changeIt );

            this->applyCreateHexahedronFunction( ea->getNbAddedHexahedra(), ea->hexahedronArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::HEXAHEDRAREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const HexahedraRemoved *>( *changeIt ) )->getArray();

            this->applyDestroyHexahedronFunction( tab );
            break;
        }
        case core::topology::HEXAHEDRAMOVED_REMOVING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const HexahedraMoved_Removing *>( *changeIt ) )->edgesAroundVertexMoved;
            container_type& data = *(this->beginEdit());

            for (unsigned int i = 0; i <edgeList.size(); i++)
                m_destroyFunc( edgeList[i], m_destroyParam, data[edgeList[i]] );

            this->endEdit();
            */
            break;
        }
        case core::topology::HEXAHEDRAMOVED_ADDING:
        {
            /*
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const HexahedraMoved_Adding *>( *changeIt ) )->edgesAroundVertexMoved;
            const sofa::helper::vector< Edge >& edgeArray = ( static_cast< const HexahedraMoved_Adding *>( *changeIt ) )->edgeArray2Moved;
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
            */
            break;
        }
        case core::topology::HEXAHEDRARENUMBERING:
        {
            const sofa::helper::vector<unsigned int> tab = ( static_cast< const HexahedraRenumbering * >( *changeIt ) )->getIndexArray();

            this->renumber( tab );
            break;
        }
        default:
            break;
        }; // switch( changeType )

        ++changeIt;
    }
}



///////////////////// Public functions to call pointer to fonction ////////////////////////
/// Apply adding points elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyCreatePointFunction(const sofa::helper::vector<unsigned int> &indices, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
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
void TopologyData <TopologyElementType, VecT>::applyCreateEdgeFunction(const sofa::helper::vector<unsigned int> &indices, const sofa::helper::vector<TopologyElementType> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
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
void TopologyData <TopologyElementType, VecT>::applyCreateTriangleFunction(const sofa::helper::vector<unsigned int> &indices, const sofa::helper::vector<TopologyElementType> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
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
void TopologyData <TopologyElementType, VecT>::applyCreateQuadFunction(const sofa::helper::vector<unsigned int> &indices, const sofa::helper::vector<TopologyElementType> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
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
void TopologyData <TopologyElementType, VecT>::applyCreateTetrahedronFunction(const sofa::helper::vector<unsigned int> &indices, const sofa::helper::vector<TopologyElementType> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
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
void TopologyData <TopologyElementType, VecT>::applyCreateHexahedronFunction(const sofa::helper::vector<unsigned int> &indices, const sofa::helper::vector<TopologyElementType> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    if (m_createHexahedronFunc)
    {
        (*m_createHexahedronFunc)(indices,m_createParam,*(this->beginEdit() ) );
        this->endEdit();
    }
}
/// Apply removing hexahedra elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::applyDestroyHexahedronFunction(const sofa::helper::vector<unsigned int> &indices)
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
}
/// Destruction function, called when deleting elements.
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::setDestroyFunction(t_destroyFunc destroyFunc)
{
    m_destroyFunc=destroyFunc;
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
        const sofa::helper::vector<TopologyElementType> &elems,
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
            this->m_createFunc( i0+i, m_createParam, t, elems[i], empty_vecint, empty_vecdouble);
        }
        else
            this->m_createFunc( i0+i, m_createParam, t, elems[i], ancestors[i], coefs[i] );
    }
    this->endEdit();
}


template <typename VecT>
void TopologyData <TopologyElementType, VecT>::move( const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    container_type& data = *(this->beginEdit());

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        this->m_destroyFunc( indexList[i], m_destroyParam, data[indexList[i]] );
        this->m_createFunc( indexList[i], m_createParam, data[indexList[i]], ancestors[i], coefs[i] );
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
        this->m_destroyFunc( index[i], m_destroyParam, data[index[i]] );
        this->swap( index[i], last );
        --last;
    }

    data.resize( data.size() - index.size() );
    this->endEdit();
    //sout << "EdgeData: vector has now "<<this->size()<<" entries."<<sendl;
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::renumber( const sofa::helper::vector<unsigned int> &index )
{
    container_type& data = *(this->beginEdit());

    container_type copy = this->getValue(); // not very efficient memory-wise, but I can see no better solution...
    for (unsigned int i = 0; i < index.size(); ++i)
        data[i] = copy[ index[i] ];

    this->endEdit();
}



///////////////////// Specializatio for Point /////////////////////////////

template< typename VecT >
void PointDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new PointSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< typename VecT >
void PointDataNew<VecT>::applyCreatePointFunction(unsigned int nbPoints, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( nbPoints, ancestors, coefs );
}


template< typename VecT >
void PointDataNew<VecT>::applyDestroyPointFunction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}



///////////////////// Specializatio for Edge /////////////////////////////

template< typename VecT >
void EdgeDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new EdgeSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< typename VecT >
void EdgeDataNew<VecT>::applyCreateEdgeFunction(unsigned int nbEdges, const sofa::helper::vector<Edge> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( nbEdges, elems, ancestors, coefs );
}


template< typename VecT >
void EdgeDataNew<VecT>::applyDestroyEdgeFunction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}



///////////////////// Specializatio for Triangle /////////////////////////////

template< typename VecT >
void TriangleDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new TriangleSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< typename VecT >
void TriangleDataNew<VecT>::applyCreateTriangleFunction(unsigned int nbTriangles, const sofa::helper::vector<Triangle> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( nbTriangles, elems, ancestors, coefs );
}


template< typename VecT >
void TriangleDataNew<VecT>::applyDestroyTriangleFunction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}



///////////////////// Specializatio for Quad /////////////////////////////

template< typename VecT >
void QuadDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new QuadSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< typename VecT >
void QuadDataNew<VecT>::applyCreateQuadFunction(unsigned int nbQuads, const sofa::helper::vector<Quad> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( nbQuads, elems, ancestors, coefs );
}


template< typename VecT >
void QuadDataNew<VecT>::applyDestroyQuadFunction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}



///////////////////// Specializatio for Tetrahedron /////////////////////////////

template< typename VecT >
void TetrahedronDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new TetrahedronSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< typename VecT >
void TetrahedronDataNew<VecT>::applyCreateTetrahedronFunction(unsigned int nbTetrahedra, const sofa::helper::vector<Tetrahedron> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( nbTetrahedra, elems, ancestors, coefs );
}


template< typename VecT >
void TetrahedronDataNew<VecT>::applyDestroyTetrahedronFunction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}



///////////////////// Specializatio for Hetrahedron /////////////////////////////

template< typename VecT >
void HexahedronDataNew<VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    if (_topology)
        this->m_topologicalEngine = new HexahedronSetTopologyEngine<VecT>((sofa::core::topology::BaseTopologyData<VecT>*)this, _topology);
#else
    (void)_topology;
#endif
}


template< typename VecT >
void HexahedronDataNew<VecT>::applyCreateHexahedronFunction(unsigned int nbHexahedra, const sofa::helper::vector<Hexahedron> &elems, const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors, const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( nbHexahedra, elems, ancestors, coefs );
}


template< typename VecT >
void HexahedronDataNew<VecT>::applyDestroyHexahedronFunction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}



} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
