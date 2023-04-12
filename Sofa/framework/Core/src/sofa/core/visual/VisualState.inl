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

#include <sofa/core/visual/VisualState.h>

namespace sofa::core::visual
{

template< typename DataTypes >
VisualState<DataTypes>::VisualState()
    : m_positions(initData(&m_positions, "position", "Vertices coordinates"))
    , m_restPositions(initData(&m_restPositions, "restPosition", "Vertices rest coordinates"))
    , m_vnormals(initData(&m_vnormals, "normal", "Normals of the model"))
    , modified(false)
{
    m_positions.setGroup("Vector");
    m_restPositions.setGroup("Vector");
    m_vnormals.setGroup("Vector");
}

template< typename DataTypes >
void VisualState<DataTypes>::resize(Size vsize)
{
    helper::WriteOnlyAccessor< Data<VecCoord > > positions = m_positions;
    if (positions.size() == vsize) return;
    helper::WriteOnlyAccessor< Data<VecCoord > > restPositions = m_restPositions;
    helper::WriteOnlyAccessor< Data<VecDeriv > > normals = m_vnormals;

    positions.resize(vsize);
    restPositions.resize(vsize); // todo allocate restpos only when it is necessary
    normals.resize(vsize);

    modified = true;
}

template< typename DataTypes >
auto VisualState<DataTypes>::write(core::VecCoordId  v) -> Data<VisualState::VecCoord>*
{
    modified = true;

    if (v == core::VecCoordId::position())
        return &m_positions;
    if (v == core::VecCoordId::restPosition())
        return &m_restPositions;

    return nullptr;
}

template< typename DataTypes >
auto VisualState<DataTypes>::read(core::ConstVecCoordId  v)  const -> const Data<VisualState::VecCoord>*
{
    if (v == core::VecCoordId::position())
        return &m_positions;
    if (v == core::VecCoordId::restPosition())
        return &m_restPositions;

    return nullptr;
}

template< typename DataTypes >
auto VisualState<DataTypes>::write(core::VecDerivId v) -> Data<VisualState::VecDeriv>*
{
    if (v == core::VecDerivId::normal())
        return &m_vnormals;

    return nullptr;
}

template< typename DataTypes >
auto VisualState<DataTypes>::read(core::ConstVecDerivId v) const -> const Data<VisualState::VecDeriv>*
{
    if (v == core::VecDerivId::normal())
        return &m_vnormals;

    return nullptr;
}

} // namespace sofa::core::visual
