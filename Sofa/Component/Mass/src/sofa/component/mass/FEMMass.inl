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
#include <sofa/component/mass/FEMMass.h>
#include <sofa/component/mass/BaseFEMMassMatrix.inl>

namespace sofa::component::mass
{

template <class TDataTypes, class TElementType>
FEMMass<TDataTypes, TElementType>::FEMMass()
    : l_nodalMassDensity(initLink("nodalMassDensity", "Link to nodal mass density"))
{
}

template <class TDataTypes, class TElementType>
auto FEMMass<TDataTypes, TElementType>::getNodalCoefficient() const -> const NodalCoefficient*
{
    return l_nodalMassDensity.get();
}

template <class TDataTypes, class TElementType>
void FEMMass<TDataTypes, TElementType>::validateNodalCoefficient()
{
    if (l_nodalMassDensity.empty())
    {
        msg_info() << "Link to a nodal mass density should be set to ensure right behavior. First "
                      "component found in current context will be used.";
        l_nodalMassDensity.set(this->getContext()->template get<NodalMassDensity>());
    }

    if (l_nodalMassDensity == nullptr)
    {
        msg_error() << "No nodal mass density component found at path: " << this->l_nodalMassDensity.getLinkedPath()
                    << ", nor in current context: " << this->getContext()->name << ".";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

template <class TDataTypes, class TElementType>
void FEMMass<TDataTypes, TElementType>::addForce(const core::MechanicalParams* mparams,
                                                 sofa::DataVecDeriv_t<DataTypes>& f,
                                                 const sofa::DataVecCoord_t<DataTypes>& x,
                                                 const sofa::DataVecDeriv_t<DataTypes>& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);
    SOFA_UNUSED(v);

    if (this->isComponentStateInvalid())
        return;

    auto forceAccessor = sofa::helper::getWriteAccessor(f);

    const auto g = this->getContext()->getGravity();
    Deriv_t<DataTypes> theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2] );

    const auto& matrix = this->m_globalMatrix;
    for (std::size_t xi = 0; xi < matrix.rowIndex.size(); ++xi)
    {
        const auto rowId = matrix.rowIndex[xi];
        typename GlobalMatrixType::Range rowRange(matrix.rowBegin[xi], matrix.rowBegin[xi + 1]);
        for (typename GlobalMatrixType::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto& value = matrix.colsValue[xj];
            forceAccessor[rowId] += value * theGravity;
        }
    }
}

template <class TDataTypes, class TElementType>
SReal FEMMass<TDataTypes, TElementType>::getPotentialEnergy(
    const core::MechanicalParams* mparams,
    const DataVecCoord_t<DataTypes>& x) const
{
    SOFA_UNUSED(mparams);

    if (this->isComponentStateInvalid())
        return 0_sreal;

    SReal potentialEnergy = 0.0;
    auto xAccessor = sofa::helper::getReadAccessor(x);

    const auto g = this->getContext()->getGravity();
    Deriv_t<DataTypes> theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2] );

    const auto& matrix = this->m_globalMatrix;
    for (std::size_t xi = 0; xi < matrix.rowIndex.size(); ++xi)
    {
        const auto rowId = matrix.rowIndex[xi];
        typename GlobalMatrixType::Range rowRange(matrix.rowBegin[xi], matrix.rowBegin[xi + 1]);
        for (typename GlobalMatrixType::Index xj = rowRange.begin(); xj < rowRange.end(); ++xj)
        {
            const auto& value = matrix.colsValue[xj];
            potentialEnergy -= sofa::type::dot(xAccessor[rowId], theGravity * value);
        }
    }

    return potentialEnergy;
}

}  // namespace sofa::component::mass
