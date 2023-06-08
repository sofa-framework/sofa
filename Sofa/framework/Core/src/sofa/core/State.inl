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

#include <sofa/core/State.h>
#include <sofa/core/AccumulationVecId.inl>

namespace sofa::core
{
template <class TDataTypes>
void State<TDataTypes>::addToTotalForces(core::ConstVecDerivId forceId)
{
    accumulatedForces.addToContributingVecIds(forceId);
}

template <class TDataTypes>
void State<TDataTypes>::removeFromTotalForces(core::ConstVecDerivId forceId)
{
    accumulatedForces.removeFromContributingVecIds(forceId);
}

template <class TDataTypes>
State<TDataTypes>::State()
    : accumulatedForces(*this)
{
    State::addToTotalForces(core::ConstVecDerivId::force());
}

template<class DataTypes>
objectmodel::BaseData* State<DataTypes>::baseWrite(VecId v)
{
    switch (v.getType())
    {
    case V_ALL: return nullptr;
    case V_COORD: return write(VecCoordId(v));
    case V_DERIV: return write(VecDerivId(v));
    case V_MATDERIV: return write(MatrixDerivId(v));
    }
    return nullptr;
}

template<class DataTypes>
const objectmodel::BaseData* State<DataTypes>::baseRead(ConstVecId v) const
{
    switch (v.getType())
    {
    case V_ALL: return nullptr;
    case V_COORD: return read(ConstVecCoordId(v));
    case V_DERIV: return read(ConstVecDerivId(v));
    case V_MATDERIV: return read(ConstMatrixDerivId(v));
    }
    return nullptr;
}

template<class DataTypes>
auto State<DataTypes>::computeBBox() const -> sofa::type::TBoundingBox<Real>
{
    const VecCoord& x = read(ConstVecCoordId::position())->getValue();
    const size_t xSize = x.size();

    if (xSize <= 0)
        return {};

    Real p[3];
    DataTypes::get(p[0], p[1], p[2], x[0]);
    Real maxBBox[3] = {p[0], p[1], p[2]};
    Real minBBox[3] = {p[0], p[1], p[2]};

    for (size_t i = 1; i < xSize; i++)
    {
        DataTypes::get(p[0], p[1], p[2], x[i]);
        for (int c = 0; c < 3; c++)
        {
            if (p[c] > maxBBox[c])
                maxBBox[c] = p[c];

            else if (p[c] < minBBox[c])
                minBBox[c] = p[c];
        }
    }

    return sofa::type::TBoundingBox<Real>(minBBox,maxBBox);
}

template<class DataTypes>
void State<DataTypes>::computeBBox(const core::ExecParams*, bool)
{
    this->f_bbox.setValue(computeBBox());
}
} // namespace sofa::core
