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

#include <sofa/component/topology/PointSetTopologyChange.h>
#include <sofa/component/topology/EdgeSetTopologyChange.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>
#include <sofa/component/topology/QuadSetTopologyChange.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>


namespace sofa
{

namespace component
{

namespace topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::handleTopologyEvents( std::list< const core::topology::TopologyChange *>::const_iterator changeIt,
        std::list< const core::topology::TopologyChange *>::const_iterator &end )
{
    if (!m_topologyData)
        return;

    while( changeIt != end )
    {
        core::topology::TopologyChangeType changeType = (*changeIt)->getChangeType();

        switch( changeType )
        {
            ///////////////////////// Events on Points //////////////////////////////////////
        case core::topology::POINTSINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const PointsIndicesSwap* >( *changeIt ) )->index[1];

            this->applyPointIndicesSwap(i1, i2);
            break;
        }
        case core::topology::POINTSADDED:
        {
            const sofa::helper::vector< unsigned int >& indexList = ( static_cast< const PointsAdded * >( *changeIt ) )->pointIndexArray;
            sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = ( static_cast< const PointsAdded * >( *changeIt ) )->ancestorsList;
            sofa::helper::vector< sofa::helper::vector< double       > >& coefs     = ( static_cast< const PointsAdded * >( *changeIt ) )->coefs;

            this->applyPointCreation(indexList, ancestors, coefs);
            break;
        }
        case core::topology::POINTSREMOVED:
        {
            const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRemoved * >( *changeIt ) )->getArray();

            this->applyPointDestruction( tab );
            break;
        }
        case core::topology::POINTSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int>& tab = ( static_cast< const PointsRenumbering * >( *changeIt ) )->getIndexArray();

            this->applyPointRenumbering(tab);
            break;
        }
        case core::topology::POINTSMOVED:
        {
            const sofa::helper::vector< unsigned int >& indexList = ( static_cast< const PointsMoved * >( *changeIt ) )->indicesList;
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = ( static_cast< const PointsMoved * >( *changeIt ) )->ancestorsList;
            sofa::helper::vector< sofa::helper::vector< double > >& coefs = ( static_cast< const PointsMoved * >( *changeIt ) )->baryCoefsList;

            this->applyPointMove( indexList, ancestors, coefs);
            break;
        }

        ///////////////////////// Events on Edges //////////////////////////////////////
        case core::topology::EDGESINDICESSWAP:
        {
            unsigned int i1 = ( static_cast< const EdgesIndicesSwap * >( *changeIt ) )->index[0];
            unsigned int i2 = ( static_cast< const EdgesIndicesSwap* >( *changeIt ) )->index[1];

            this->applyEdgeIndicesSwap( i1, i2 );
            break;
        }
        case core::topology::EDGESADDED:
        {
            const EdgesAdded *ea=static_cast< const EdgesAdded * >( *changeIt );

            this->applyEdgeCreation( ea->edgeIndexArray, ea->edgeArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::EDGESREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const EdgesRemoved *>( *changeIt ) )->getArray();

            this->applyEdgeDestruction( tab );
            break;
        }
        case core::topology::EDGESMOVED_REMOVING:
        {
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const EdgesMoved_Removing *>( *changeIt ) )->edgesAroundVertexMoved;

            this->applyEdgeMovedDestruction(edgeList);
            break;
        }
        case core::topology::EDGESMOVED_ADDING:
        {
            const sofa::helper::vector< unsigned int >& edgeList = ( static_cast< const EdgesMoved_Adding *>( *changeIt ) )->edgesAroundVertexMoved;
            const sofa::helper::vector< Edge >& edgeArray = ( static_cast< const EdgesMoved_Adding *>( *changeIt ) )->edgeArray2Moved;

            this->applyEdgeMovedCreation(edgeList, edgeArray);
            break;
        }
        case core::topology::EDGESRENUMBERING:
        {
            const sofa::helper::vector<unsigned int>& tab = ( static_cast< const EdgesRenumbering * >( *changeIt ) )->getIndexArray();

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

            this->applyTriangleCreation( ea->getNbAddedTriangles(), ea->triangleArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::TRIANGLESREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TrianglesRemoved *>( *changeIt ) )->getArray();

            this->applyTriangleDestruction( tab );
            break;
        }
        case core::topology::TRIANGLESMOVED_REMOVING:
        {
            const sofa::helper::vector< unsigned int >& triangleList = ( static_cast< const TrianglesMoved_Removing *>( *changeIt ) )->trianglesAroundVertexMoved;

            this->applyTriangleMovedDestruction(triangleList);
            break;
        }
        case core::topology::TRIANGLESMOVED_ADDING:
        {
            const sofa::helper::vector< unsigned int >& triangleList = ( static_cast< const TrianglesMoved_Adding *>( *changeIt ) )->trianglesAroundVertexMoved;
            const sofa::helper::vector< Triangle >& triangleArray = ( static_cast< const TrianglesMoved_Adding *>( *changeIt ) )->triangleArray2Moved;

            this->applyTriangleMovedCreation(triangleList, triangleArray);
            break;
        }
        case core::topology::TRIANGLESRENUMBERING:
        {
            const sofa::helper::vector<unsigned int>& tab = ( static_cast< const TrianglesRenumbering * >( *changeIt ) )->getIndexArray();

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

            this->applyQuadCreation( ea->getNbAddedQuads(), ea->quadArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::QUADSREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const QuadsRemoved *>( *changeIt ) )->getArray();

            this->applyQuadDestruction( tab );
            break;
        }
        case core::topology::QUADSMOVED_REMOVING:
        {
            const sofa::helper::vector< unsigned int >& quadList = ( static_cast< const QuadsMoved_Removing *>( *changeIt ) )->quadsAroundVertexMoved;

            this->applyQuadMovedDestruction(quadList);
            break;
        }
        case core::topology::QUADSMOVED_ADDING:
        {
            const sofa::helper::vector< unsigned int >& quadList = ( static_cast< const QuadsMoved_Adding *>( *changeIt ) )->quadsAroundVertexMoved;
            const sofa::helper::vector< Triangle >& quadArray = ( static_cast< const QuadsMoved_Adding *>( *changeIt ) )->quadArray2Moved;

            this->applyQuadMovedCreation(quadList, quadArray);
            break;
        }
        case core::topology::QUADSRENUMBERING:
        {
            const sofa::helper::vector<unsigned int>& tab = ( static_cast< const QuadsRenumbering * >( *changeIt ) )->getIndexArray();

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

            this->applyTetrahedronCreation( ea->getNbAddedTetrahedra(), ea->tetrahedronArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::TETRAHEDRAREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const TetrahedraRemoved *>( *changeIt ) )->getArray();

            this->applyTetrahedronDestruction( tab );
            break;
        }
        case core::topology::TETRAHEDRAMOVED_REMOVING:
        {
            const sofa::helper::vector< unsigned int >& tetrahedronList = ( static_cast< const TetrahedraMoved_Removing *>( *changeIt ) )->tetrahedraAroundVertexMoved;

            this->applyTetrahedronMovedDestruction(tetrahedronList);
            break;
        }
        case core::topology::TETRAHEDRAMOVED_ADDING:
        {
            const sofa::helper::vector< unsigned int >& tetrahedronList = ( static_cast< const TetrahedraMoved_Adding *>( *changeIt ) )->tetrahedraAroundVertexMoved;
            const sofa::helper::vector< Triangle >& tetrahedronArray = ( static_cast< const TetrahedraMoved_Adding *>( *changeIt ) )->tetrahedronArray2Moved;

            this->applyTetrahedronMovedCreation(tetrahedronList, tetrahedronArray);
            break;
        }
        case core::topology::TETRAHEDRARENUMBERING:
        {
            const sofa::helper::vector<unsigned int>& tab = ( static_cast< const TetrahedraRenumbering * >( *changeIt ) )->getIndexArray();

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

            this->applyHexahedronCreation( ea->getNbAddedHexahedra(), ea->hexahedronArray, ea->ancestorsList, ea->coefs );
            break;
        }
        case core::topology::HEXAHEDRAREMOVED:
        {
            const sofa::helper::vector<unsigned int> &tab = ( static_cast< const HexahedraRemoved *>( *changeIt ) )->getArray();

            this->applyHexahedronDestruction( tab );
            break;
        }
        case core::topology::HEXAHEDRAMOVED_REMOVING:
        {
            const sofa::helper::vector< unsigned int >& hexahedronList = ( static_cast< const HexahedraMoved_Removing *>( *changeIt ) )->hexahedraAroundVertexMoved;

            this->applyHexahedronMovedDestruction(hexahedronList);
            break;
        }
        case core::topology::HEXAHEDRAMOVED_ADDING:
        {
            const sofa::helper::vector< unsigned int >& hexahedronList = ( static_cast< const HexahedraMoved_Adding *>( *changeIt ) )->hexahedraAroundVertexMoved;
            const sofa::helper::vector< Triangle >& hexahedronArray = ( static_cast< const HexahedraMoved_Adding *>( *changeIt ) )->hexahedronArray2Moved;

            this->applyHexahedronMovedCreation(hexahedronList, hexahedronArray);
            break;
        }
        case core::topology::HEXAHEDRARENUMBERING:
        {
            const sofa::helper::vector<unsigned int>& tab = ( static_cast< const HexahedraRenumbering * >( *changeIt ) )->getIndexArray();

            this->renumber( tab );
            break;
        }
        default:
            break;
        }; // switch( changeType )

        ++changeIt;
    }
}



///////////////////// Private functions on TopologyDataImpl changes /////////////////////////////

template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::swap( unsigned int i1, unsigned int i2 )
{
    container_type& data = *(m_topologyData->beginEdit());
    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;
    m_topologyData->endEdit();
}

template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::add(unsigned int nbElements,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    // Using default values
    container_type& data = *(m_topologyData->beginEdit());
    unsigned int i0 = data.size();
    data.resize(i0+nbElements);

    for (unsigned int i = 0; i < nbElements; ++i)
    {
        value_type& t = data[i0+i];
        if (ancestors.empty() || coefs.empty())
        {
            const sofa::helper::vector< unsigned int > empty_vecint;
            const sofa::helper::vector< double > empty_vecdouble;
            this->applyCreateFunction( i0+i, t, empty_vecint, empty_vecdouble);
        }
        else
            this->applyCreateFunction( i0+i, t, ancestors[i], coefs[i] );
    }
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::add(unsigned int nbElements,
        const sofa::helper::vector< TopologyElementType >& elems,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    // Using default values
    container_type& data = *(m_topologyData->beginEdit());
    unsigned int i0 = data.size();
    data.resize(i0+nbElements);

    for (unsigned int i = 0; i < nbElements; ++i)
    {
        value_type& t = data[i0+i];
        if (ancestors.empty() || coefs.empty())
        {
            const sofa::helper::vector< unsigned int > empty_vecint;
            const sofa::helper::vector< double > empty_vecdouble;
            this->applyCreateFunction( i0+i, t, elems[i], empty_vecint, empty_vecdouble);
        }
        else
            this->applyCreateFunction( i0+i, t, elems[i], ancestors[i], coefs[i] );
    }
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::move( const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        this->applyDestroyFunction( indexList[i], data[indexList[i]] );
        this->applyCreateFunction( indexList[i], data[indexList[i]], ancestors[i], coefs[i] );
    }

    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::remove( const sofa::helper::vector<unsigned int> &index )
{

    container_type& data = *(m_topologyData->beginEdit());
    unsigned int last = data.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        this->applyDestroyFunction( index[i], data[index[i]] );
        this->swap( index[i], last );
        --last;
    }

    data.resize( data.size() - index.size() );
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataImpl <TopologyElementType, VecT>::renumber( const sofa::helper::vector<unsigned int> &index )
{
    container_type& data = *(m_topologyData->beginEdit());

    container_type copy = this->getValue(); // not very efficient memory-wise, but I can see no better solution...
    for (unsigned int i = 0; i < index.size(); ++i)
        data[i] = copy[ index[i] ];

    m_topologyData->endEdit();
}




////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////   Point Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void PointDataImpl<VecT>::applyPointCreation(const sofa::helper::vector< unsigned int >& indices,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{

    this->add(indices.size(), ancestors, coefs );
}

template< typename VecT >
void PointDataImpl<VecT>::applyPointDestruction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void PointDataImpl<VecT>::applyPointIndicesSwap(unsigned int i1, unsigned int i2)
{
    this->swap( i1, i2 );
}

template< typename VecT >
void PointDataImpl<VecT>::applyPointRenumbering(const sofa::helper::vector<unsigned int> &indices)
{
    this->renumber( indices );
}

template< typename VecT >
void PointDataImpl<VecT>::applyPointMove(const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector< Point >& /*elems*/,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->move( indexList, ancestors, coefs);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Edge Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void EdgeDataImpl<VecT>::applyEdgeCreation(const sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< Edge >& elems,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

template< typename VecT >
void EdgeDataImpl<VecT>::applyEdgeDestruction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void EdgeDataImpl<VecT>::applyEdgeIndicesSwap(unsigned int i1, unsigned int i2)
{
    this->swap( i1, i2 );
}

template< typename VecT >
void EdgeDataImpl<VecT>::applyeEdgeRenumbering(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void EdgeDataImpl<VecT>::applyEdgeMovedCreation(const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector<Edge> & edgeArray)
{
    container_type& data = *(this->beginEdit());

    // Recompute data
    sofa::helper::vector< unsigned int > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back (1.0);
    ancestors.resize(1);

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        ancestors[0] = indexList[i];
        this->applyCreateFunction( indexList[i], data[indexList[i]], edgeArray[i], ancestors, coefs );
    }

    this->endEdit();
}

template< typename VecT >
void EdgeDataImpl<VecT>::applyEdgeMovedDestruction(const sofa::helper::vector<unsigned int> &indices)
{

    container_type& data = *(this->beginEdit());

    for (unsigned int i = 0; i <indices.size(); i++)
        this->applyDestroyFunction( indices[i], data[indices[i]] );


    this->endEdit();

    this->remove( indices );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Triangle Topology Data Implementation   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void TriangleDataImpl<VecT>::applyTriangleCreation(const sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< Triangle >& elems,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

template< typename VecT >
void TriangleDataImpl<VecT>::applyTriangleDestruction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void TriangleDataImpl<VecT>::applyTriangleIndicesSwap(unsigned int i1, unsigned int i2)
{
    this->swap( i1, i2 );
}

template< typename VecT >
void TriangleDataImpl<VecT>::applyeTriangleRenumbering(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void TriangleDataImpl<VecT>::applyTriangleMovedCreation(const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector<Triangle> & triangleArray)
{
    container_type& data = *(m_topologyData->beginEdit());

    // Recompute data
    sofa::helper::vector< unsigned int > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back (1.0);
    ancestors.resize(1);

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        ancestors[0] = triangleArray[i];
        this->applyCreateFunction( indexList[i], data[indexList[i]], triangleArray[i], ancestors, coefs );
    }

    m_topologyData->endEdit();
}

template< typename VecT >
void TriangleDataImpl<VecT>::applyTriangleMovedDestruction(const sofa::helper::vector<unsigned int> &indices)
{

    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <indices.size(); i++)
        this->applyDestroyFunction( indices[i], data[indices[i]] );


    m_topologyData->endEdit();

    this->remove( indices );
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Quad Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void QuadDataImpl<VecT>::applyQuadCreation(const sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< Quad >& elems,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

template< typename VecT >
void QuadDataImpl<VecT>::applyQuadDestruction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void QuadDataImpl<VecT>::applyQuadIndicesSwap(unsigned int i1, unsigned int i2)
{
    this->swap( i1, i2 );
}

template< typename VecT >
void QuadDataImpl<VecT>::applyeQuadRenumbering(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void QuadDataImpl<VecT>::applyQuadMovedCreation(const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector<Quad> & quadArray)
{
    container_type& data = *(m_topologyData->beginEdit());

    // Recompute data
    sofa::helper::vector< unsigned int > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back (1.0);
    ancestors.resize(1);

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        ancestors[0] = indexList[i];
        this->applyCreateFunction( indexList[i], data[indexList[i]], quadArray[i], ancestors, coefs );
    }

    m_topologyData->endEdit();
}

template< typename VecT >
void QuadDataImpl<VecT>::applyQuadMovedDestruction(const sofa::helper::vector<unsigned int> &indices)
{

    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <indices.size(); i++)
        this->applyDestroyFunction( indices[i], data[indices[i]] );


    m_topologyData->endEdit();

    this->remove( indices );
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Tetrahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void TetrahedronDataImpl<VecT>::applyTetrahedronCreation(const sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< Tetrahedron >& elems,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

template< typename VecT >
void TetrahedronDataImpl<VecT>::applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void TetrahedronDataImpl<VecT>::applyTetrahedronIndicesSwap(unsigned int i1, unsigned int i2)
{
    this->swap( i1, i2 );
}

template< typename VecT >
void TetrahedronDataImpl<VecT>::applyeTetrahedronRenumbering(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void TetrahedronDataImpl<VecT>::applyTetrahedronMovedCreation(const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector<Tetrahedron> & tetrahedronArray)
{
    container_type& data = *(m_topologyData->beginEdit());

    // Recompute data
    sofa::helper::vector< unsigned int > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back (1.0);
    ancestors.resize(1);

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        ancestors[0] = indexList[i];
        this->applyCreateFunction( indexList[i], data[indexList[i]], tetrahedronArray[i], ancestors, coefs );
    }

    m_topologyData->endEdit();
}

template< typename VecT >
void TetrahedronDataImpl<VecT>::applyTetrahedronMovedDestruction(const sofa::helper::vector<unsigned int> &indices)
{

    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <indices.size(); i++)
        this->applyDestroyFunction( indices[i], data[indices[i]] );


    m_topologyData->endEdit();

    this->remove( indices );
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Hexahedron Topology Data Implementation   /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename VecT >
void HexahedronDataImpl<VecT>::applyHexahedronCreation(const sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< Hexahedron >& elems,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

template< typename VecT >
void HexahedronDataImpl<VecT>::applyHexahedronDestruction(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void HexahedronDataImpl<VecT>::applyHexahedronIndicesSwap(unsigned int i1, unsigned int i2)
{
    this->swap( i1, i2 );
}

template< typename VecT >
void HexahedronDataImpl<VecT>::applyeHexahedronRenumbering(const sofa::helper::vector<unsigned int> &indices)
{
    this->remove( indices );
}

template< typename VecT >
void HexahedronDataImpl<VecT>::applyHexahedronMovedCreation(const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector<Hexahedron> & hexahedronArray)
{
    container_type& data = *(m_topologyData->beginEdit());

    // Recompute data
    sofa::helper::vector< unsigned int > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back (1.0);
    ancestors.resize(1);

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        ancestors[0] = indexList[i];
        this->applyCreateFunction( indexList[i], data[indexList[i]], hexahedronArray[i], ancestors, coefs );
    }

    m_topologyData->endEdit();
}

template< typename VecT >
void HexahedronDataImpl<VecT>::applyHexahedronMovedDestruction(const sofa::helper::vector<unsigned int> &indices)
{

    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <indices.size(); i++)
        this->applyDestroyFunction( indices[i], data[indices[i]] );


    m_topologyData->endEdit();

    this->remove( indices );
}





} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
