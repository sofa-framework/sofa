/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/topology/TopologyHandler.h>


namespace sofa
{

namespace core
{

namespace topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Generic Handling of Topology Event    /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TopologyHandler::ApplyTopologyChanges(const std::list<const core::topology::TopologyChange *> &_topologyChangeEvents, const unsigned int _dataSize)
{
    if(!this->isTopologyDataRegistered())
        return;

    sofa::helper::list<const core::topology::TopologyChange *>::iterator changeIt;
    sofa::helper::list<const core::topology::TopologyChange *> _changeList = _topologyChangeEvents;

    this->setDataSetArraySize(_dataSize);

    for (changeIt=_changeList.begin(); changeIt!=_changeList.end(); ++changeIt)
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
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors = ( static_cast< const PointsAdded * >( *changeIt ) )->ancestorsList;
            const sofa::helper::vector< sofa::helper::vector< double       > >& coefs     = ( static_cast< const PointsAdded * >( *changeIt ) )->coefs;

            this->applyPointCreation(indexList, indexList, ancestors, coefs);
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
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs = ( static_cast< const PointsMoved * >( *changeIt ) )->baryCoefsList;

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

            this->applyTriangleCreation( ea->triangleIndexArray, ea->triangleArray, ea->ancestorsList, ea->coefs );
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

            this->applyQuadCreation( ea->quadIndexArray, ea->quadArray, ea->ancestorsList, ea->coefs );
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
            const sofa::helper::vector< Quad >& quadArray = ( static_cast< const QuadsMoved_Adding *>( *changeIt ) )->quadArray2Moved;

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

            this->applyTetrahedronCreation( ea->tetrahedronIndexArray, ea->tetrahedronArray, ea->ancestorsList, ea->coefs );
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
            const sofa::helper::vector< Tetrahedron >& tetrahedronArray = ( static_cast< const TetrahedraMoved_Adding *>( *changeIt ) )->tetrahedronArray2Moved;

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

            this->applyHexahedronCreation( ea->hexahedronIndexArray, ea->hexahedronArray, ea->ancestorsList, ea->coefs );
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
            const sofa::helper::vector< Hexahedron >& hexahedronArray = ( static_cast< const HexahedraMoved_Adding *>( *changeIt ) )->hexahedronArray2Moved;

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

        //++changeIt;
    }
}

} // namespace topology

} // namespace core

} // namespace sofa
