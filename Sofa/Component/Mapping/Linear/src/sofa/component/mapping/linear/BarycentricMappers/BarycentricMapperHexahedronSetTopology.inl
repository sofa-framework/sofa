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
#pragma once
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperHexahedronSetTopology.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <array>

namespace sofa::component::mapping::linear
{

template <class In, class Out>
BarycentricMapperHexahedronSetTopology<In,Out>::BarycentricMapperHexahedronSetTopology()
    : Inherit1(nullptr, nullptr)
{}

template <class In, class Out>
BarycentricMapperHexahedronSetTopology<In,Out>::BarycentricMapperHexahedronSetTopology(sofa::core::topology::TopologyContainer* fromTopology,
    core::topology::BaseMeshTopology* toTopology)
    : Inherit1(fromTopology, toTopology)
{}

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::setTopology(sofa::core::topology::TopologyContainer* topology)
{
    m_fromTopology  = topology;
}


template <class In, class Out>
typename BarycentricMapperHexahedronSetTopology<In, Out>::Index 
BarycentricMapperHexahedronSetTopology<In,Out>::addPointInCube ( const Index cubeIndex, const SReal* baryCoords )
{
    auto vectorData = sofa::helper::getWriteAccessor(d_map);
    MappingData data;
    data.in_index = cubeIndex;
    data.baryCoords[0] = static_cast<Real>(baryCoords[0]);
    data.baryCoords[1] = static_cast<Real>(baryCoords[1]);
    data.baryCoords[2] = static_cast<Real>(baryCoords[2]);
    vectorData->emplace_back(data);
    return static_cast<Index>(vectorData.size() - 1u);
}


template <class In, class Out>
typename BarycentricMapperHexahedronSetTopology<In, Out>::Index 
BarycentricMapperHexahedronSetTopology<In,Out>::setPointInCube ( const Index pointIndex, const Index cubeIndex, const SReal* baryCoords )
{
    auto vectorData = sofa::helper::getWriteAccessor(d_map);

    if ( pointIndex >= vectorData.size() )
        return sofa::InvalidID;

    MappingData& data = vectorData[pointIndex];
    data.in_index = cubeIndex;
    data.baryCoords[0] = static_cast<Real>(baryCoords[0]);
    data.baryCoords[1] = static_cast<Real>(baryCoords[1]);
    data.baryCoords[2] = static_cast<Real>(baryCoords[2]);

    if(cubeIndex == sofa::InvalidID)
        m_invalidIndex.insert(pointIndex);
    else
        m_invalidIndex.erase(pointIndex);

    return pointIndex;
}


template <class In, class Out>
type::vector<Hexahedron> BarycentricMapperHexahedronSetTopology<In,Out>::getElements()
{
    return this->m_fromTopology->getHexahedra();
}


template <class In, class Out>
type::vector<SReal> BarycentricMapperHexahedronSetTopology<In,Out>::getBaryCoef(const Real* f)
{
    return getBaryCoef(f[0],f[1],f[2]);
}


template <class In, class Out>
type::vector<SReal> BarycentricMapperHexahedronSetTopology<In,Out>::getBaryCoef(const Real fx, const Real fy, const Real fz)
{
    type::vector<SReal> hexahedronCoef{(1-fx)*(1-fy)*(1-fz),
                (fx)*(1-fy)*(1-fz),
                (fx)*(fy)*(1 - fz),
                (1 - fx)*(fy)*(1 - fz),
                (1-fx)*(1-fy)*(fz),
                (fx)*(1-fy)*(fz),
                (fx)*(fy)*(fz),
                (1 - fx)*(fy)*(fz)
    };
    return hexahedronCoef;
}


template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Hexahedron& element)
{
    Mat3x3d matrixTranspose;
    base[0] = in[element[1]]-in[element[0]];
    base[1] = in[element[3]]-in[element[0]];
    base[2] = in[element[4]]-in[element[0]];
    matrixTranspose.transpose(base);
    const bool canInvert = base.invert(matrixTranspose);
    assert(canInvert);
    SOFA_UNUSED(canInvert);
}


template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::computeCenter(Vec3& center, const typename In::VecCoord& in, const Hexahedron &element)
{
    center = ( in[element[0]]+in[element[1]]+in[element[2]]+in[element[3]]+in[element[4]]+in[element[5]]+in[element[6]]+in[element[7]] ) *0.125;
}


template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::computeDistance(SReal& d, const Vec3& v)
{
    d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-1 ),std::max ( v[1]-1,v[2]-1 ) ) );
}


template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::addPointInElement(const Index elementIndex, const SReal* baryCoords)
{
    addPointInCube(elementIndex,baryCoords);
}



template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange(core::topology::Topology* t)
{
    using sofa::core::behavior::MechanicalState;

    if (t != this->m_fromTopology) return;

    if ( this->m_fromTopology->beginChange() == this->m_fromTopology->endChange() )
        return;

    const std::list<const core::topology::TopologyChange *>::const_iterator itBegin = this->m_fromTopology->beginChange();
    const std::list<const core::topology::TopologyChange *>::const_iterator itEnd = this->m_fromTopology->endChange();

    for ( std::list<const core::topology::TopologyChange *>::const_iterator changeIt = itBegin;
            changeIt != itEnd; ++changeIt )
    {
        const core::topology::TopologyChangeType changeType = ( *changeIt )->getChangeType();
        switch ( changeType )
        {
            //TODO: implementation of BarycentricMapperHexahedronSetTopology<In,Out>::handleTopologyChange()
        case core::topology::ENDING_EVENT:       ///< To notify the end for the current sequence of topological change events
        {
            if(!m_invalidIndex.empty())
            {
                type::vector<MappingData>& mapData = *(d_map.beginEdit());

                for ( auto iter = m_invalidIndex.cbegin();
                        iter != m_invalidIndex.cend(); ++iter )
                {
                    const auto j = *iter;
                    if ( mapData[j].in_index == sofa::InvalidID ) // compute new mapping
                    {
                        type::Vec3 coefs;
                        typename In::Coord pos;
                        pos[0] = mapData[j].baryCoords[0];
                        pos[1] = mapData[j].baryCoords[1];
                        pos[2] = mapData[j].baryCoords[2];

                        // find nearest cell and barycentric coords
                        SReal distance = 1e10;

                        Index index = sofa::InvalidID;
                        // When smoothing a mesh, the element has to be found using the rest position of the point. Then, its position is set using this element.
                        typedef MechanicalState<In> InMechanicalStateT;
                        InMechanicalStateT* inState;
                        this->m_fromTopology->getContext()->get(inState);
                        const auto& inRestPos = (inState->read(core::ConstVecCoordId::restPosition())->getValue());
                        if( this->m_toTopology)
                        {
                            typedef MechanicalState<Out> MechanicalStateT;
                            MechanicalStateT* mState;
                            this->m_toTopology->getContext()->get( mState);
                            if( !mState)
                            {
                                msg_error() << "Can not find mechanical state." ;
                            }
                            else
                            {
                                const typename MechanicalStateT::VecCoord& outXto0 = (mState->read(core::ConstVecCoordId::restPosition())->getValue());
                                const decltype(inRestPos[0])& outRestPos = Out::getCPos(outXto0[j]); //decltype stuff is to force the same type of coordinates between in and out
                                index = sofa::topology::getClosestHexahedronIndex(inRestPos, m_fromTopology->getHexahedra(), outRestPos, coefs, distance);
                            }
                        }
                        else
                        {

                            index = sofa::topology::getClosestHexahedronIndex(inRestPos, m_fromTopology->getHexahedra(), pos, coefs, distance);
                        }

                        if ( index != sofa::InvalidID )
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
            typedef MechanicalState<In> InMechanicalStateT;
            InMechanicalStateT* inState;
            this->m_fromTopology->getContext()->get(inState);
            const auto& inRestPos = (inState->read(core::ConstVecCoordId::restPosition())->getValue());

            const auto nbHexahedra = this->m_fromTopology->getNbHexahedra();

            const auto &hexahedra =
                    ( static_cast< const core::topology::HexahedraRemoved *> ( *changeIt ) )->getArray();
            //        type::vector<unsigned int> hexahedra(tab);

            for ( std::size_t i=0; i<hexahedra.size(); ++i )
            {
                // remove all references to the removed cubes from the mapping data
                Index cubeId = hexahedra[i];
                for ( std::size_t j=0; j<d_map.getValue().size(); ++j )
                {
                    if (d_map.getValue()[j].in_index == cubeId) // invalidate mapping
                    {
                        const auto& baryMap = d_map.getValue()[j];
                        sofa::type::fixed_array<SReal, 3> coefs;
                        coefs[0] = baryMap.baryCoords[0];
                        coefs[1] = baryMap.baryCoords[1];
                        coefs[2] = baryMap.baryCoords[2];

                        const auto& h = this->m_fromTopology->getHexahedron(cubeId);

                        const auto restPos = sofa::geometry::Hexahedron::getPositionFromBarycentricCoefficients(inRestPos[h[0]], inRestPos[h[1]], inRestPos[h[2]], inRestPos[h[3]], 
                                                                                                                inRestPos[h[4]], inRestPos[h[5]], inRestPos[h[6]], inRestPos[h[7]], coefs);

                        type::vector<MappingData>& vectorData = *(d_map.beginEdit());
                        vectorData[j].in_index = sofa::InvalidID;
                        vectorData[j].baryCoords[0] = restPos[0];
                        vectorData[j].baryCoords[1] = restPos[1];
                        vectorData[j].baryCoords[2] = restPos[2];
                        d_map.endEdit();

                        m_invalidIndex.insert(Size(j));
                    }
                }
            }

            // renumber
            Index lastCubeId = nbHexahedra-1;
            for ( std::size_t i=0; i<hexahedra.size(); ++i, --lastCubeId )
            {
                Index cubeId = hexahedra[i];
                for (Index j=0; j<d_map.getValue().size(); ++j )
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

template <class In, class Out>
void BarycentricMapperHexahedronSetTopology<In,Out>::applyOnePoint( const Index& hexaPointId,typename Out::VecCoord& out, const typename In::VecCoord& in )
{
    const type::vector<Hexahedron>& cubes = this->m_fromTopology->getHexahedra();
    const Real fx = d_map.getValue()[hexaPointId].baryCoords[0];
    const Real fy = d_map.getValue()[hexaPointId].baryCoords[1];
    const Real fz = d_map.getValue()[hexaPointId].baryCoords[2];
    const Index index = d_map.getValue()[hexaPointId].in_index;
    const Hexahedron& cube = cubes[index];
    Out::setCPos(out[hexaPointId] , in[cube[0]] * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) )
            + in[cube[1]] * ( ( fx ) * ( 1-fy ) * ( 1-fz ) )
            + in[cube[3]] * ( ( 1-fx ) * ( fy ) * ( 1-fz ) )
            + in[cube[2]] * ( ( fx ) * ( fy ) * ( 1-fz ) )
            + in[cube[4]] * ( ( 1-fx ) * ( 1-fy ) * ( fz ) )
            + in[cube[5]] * ( ( fx ) * ( 1-fy ) * ( fz ) )
            + in[cube[7]] * ( ( 1-fx ) * ( fy ) * ( fz ) )
            + in[cube[6]] * ( ( fx ) * ( fy ) * ( fz ) ) );
}


} // namespace sofa::component::mapping::linear
