/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_MAPPING_CUDABARYCENTRICMAPPINGRIGID_CPP
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/core/Mapping.inl>

#include <sofa/component/mapping/linear/BarycentricMappingRigid.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/TopologyBarycentricMapper.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperHexahedronSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTriangleSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperQuadSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperEdgeSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperSparseGridTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperRegularGridTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTopologyContainer.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapper.inl>

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::mapping::linear
{

using namespace sofa::defaulttype;
using namespace sofa::gpu::cuda;

// Register in the Factory
int BarycentricMappingRigidClass = core::RegisterObject("")
        .add< BarycentricMapping< CudaVec3Types, Rigid3Types > >()

        ;


template <>
void BarycentricMapperHexahedronSetTopology<CudaVec3Types, defaulttype::Rigid3Types>::handleTopologyChange(core::topology::Topology* t)
{
    if (t != this->m_fromTopology) return;
    if ( this->m_fromTopology->beginChange() == this->m_fromTopology->endChange() )
        return;

    const std::list<const core::topology::TopologyChange *>::const_iterator itBegin = this->m_fromTopology->beginChange();
    const std::list<const core::topology::TopologyChange *>::const_iterator itEnd = this->m_fromTopology->endChange();

    typedef sofa::core::behavior::MechanicalState<defaulttype::Vec3Types> InMechanicalStateT;
    InMechanicalStateT* inState;
    this->m_fromTopology->getContext()->get(inState);
    const auto& inRestPos = (inState->read(core::ConstVecCoordId::restPosition())->getValue());

    for ( std::list<const core::topology::TopologyChange *>::const_iterator changeIt = itBegin;
            changeIt != itEnd; ++changeIt )
    {
        const core::topology::TopologyChangeType changeType = ( *changeIt )->getChangeType();
        switch ( changeType )
        {
        case core::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
        {
            if(!m_invalidIndex.empty())
            {
                type::vector<MappingData>& mapData = *(d_map.beginEdit());

                for ( auto iter = m_invalidIndex.cbegin();
                        iter != m_invalidIndex.cend(); ++iter )
                {
                    const int j = *iter;
                    if ( mapData[j].in_index == sofa::InvalidID ) // compute new mapping
                    {
                        sofa::type::Vec3 coefs;
                        defaulttype::Vec3Types::Coord pos;
                        pos[0] = mapData[j].baryCoords[0];
                        pos[1] = mapData[j].baryCoords[1];
                        pos[2] = mapData[j].baryCoords[2];

                        // find nearest cell and barycentric coords
                        SReal distance = 1e10;

                        const Index index = sofa::topology::getClosestHexahedronIndex(inRestPos, m_fromTopology->getHexahedra(), pos, coefs, distance);

                        if ( index != -1 )
                        {
                            mapData[j].baryCoords[0] = ( Real ) coefs[0];
                            mapData[j].baryCoords[1] = ( Real ) coefs[1];
                            mapData[j].baryCoords[2] = ( Real ) coefs[2];
                            mapData[j].in_index = index;
                        }
                    }
                }

                d_map.endEdit();
                m_invalidIndex.clear();
            }
        }
        break;
        case core::topology::POINTSINDICESSWAP:  ///< For PointsIndicesSwap.
            break;
        case core::topology::POINTSADDED:        ///< For PointsAdded.
            break;
        case core::topology::POINTSREMOVED:      ///< For PointsRemoved.
            break;
        case core::topology::POINTSRENUMBERING:  ///< For PointsRenumbering.
            break;
        case core::topology::TRIANGLESADDED:  ///< For Triangles Added.
            break;
        case core::topology::TRIANGLESREMOVED:  ///< For Triangles Removed.
            break;
        case core::topology::HEXAHEDRAADDED:     ///< For HexahedraAdded.
        {
        }
        break;
        case core::topology::HEXAHEDRAREMOVED:   ///< For HexahedraRemoved.
        {
            const unsigned int nbHexahedra = this->m_fromTopology->getNbHexahedra();

            const auto &hexahedra = ( static_cast< const core::topology::HexahedraRemoved *> ( *changeIt ) )->getArray();

            for ( std::size_t i=0; i<hexahedra.size(); ++i )
            {
                // remove all references to the removed cubes from the mapping data
                const unsigned int cubeId = hexahedra[i];
                for ( unsigned int j=0; j<d_map.getValue().size(); ++j )
                {
                    if ( d_map.getValue()[j].in_index == cubeId ) // invalidate mapping
                    {
                        sofa::type::fixed_array<SReal, 3> coefs;
                        coefs[0] = d_map.getValue()[j].baryCoords[0];
                        coefs[1] = d_map.getValue()[j].baryCoords[1];
                        coefs[2] = d_map.getValue()[j].baryCoords[2];

                        const auto& h = this->m_fromTopology->getHexahedron(cubeId);
                        const auto restPos = sofa::geometry::Hexahedron::getPositionFromBarycentricCoefficients(inRestPos[h[0]], inRestPos[h[1]], inRestPos[h[2]], inRestPos[h[3]],
                            inRestPos[h[4]], inRestPos[h[5]], inRestPos[h[6]], inRestPos[h[7]], coefs);

                        type::vector<MappingData>& vectorData = *(d_map.beginEdit());
                        vectorData[j].in_index = sofa::InvalidID;
                        vectorData[j].baryCoords[0] = restPos[0];
                        vectorData[j].baryCoords[1] = restPos[1];
                        vectorData[j].baryCoords[2] = restPos[2];
                        d_map.endEdit();

                        m_invalidIndex.insert(j);
                    }
                }
            }

            // renumber
            unsigned int lastCubeId = nbHexahedra-1;
            for ( unsigned int i=0; i<hexahedra.size(); ++i, --lastCubeId )
            {
                const unsigned int cubeId = hexahedra[i];
                for ( unsigned int j=0; j<d_map.getValue().size(); ++j )
                {
                    if ( d_map.getValue()[j].in_index == lastCubeId )
                    {
                        type::vector<MappingData>& vectorData = *(d_map.beginEdit());
                        vectorData[j].in_index = cubeId;
                        d_map.endEdit();
                    }
                }
            }
        }
        break;
        case core::topology::HEXAHEDRARENUMBERING: ///< For HexahedraRenumbering.
            break;
        default:
            break;
        }
    }
}



template class SOFA_GPU_CUDA_API BarycentricMapping< CudaVec3Types, Rigid3Types >;

} // namespace sofa::component::mapping::linear
