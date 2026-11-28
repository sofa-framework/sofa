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
#include <sofa/component/mass/FEMCapacity.h>
#include <sofa/component/mass/BaseFEMMassMatrix.inl>

namespace sofa::component::mass
{

template <class TDataTypes, class TSpatialTypes, class TElementType>
FEMCapacity<TDataTypes, TSpatialTypes, TElementType>::FEMCapacity()
    : l_nodalCapacity(initLink("nodalCapacity", "Link to nodal capacity"))
{
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
auto FEMCapacity<TDataTypes, TSpatialTypes, TElementType>::getNodalCoefficient() const -> const NodalCoefficient*
{
    return l_nodalCapacity.get();
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void FEMCapacity<TDataTypes, TSpatialTypes, TElementType>::validateNodalCoefficient()
{
    if (l_nodalCapacity.empty())
    {
        msg_info() << "Link to a nodal capacity should be set to ensure right behavior. First "
                      "component found in current context will be used.";
        l_nodalCapacity.set(this->getContext()->template get<NodalCapacity>());
    }

    if (l_nodalCapacity == nullptr)
    {
        msg_error() << "No nodal capacity component found at path: " << this->l_nodalCapacity.getLinkedPath()
                    << ", nor in current context: " << this->getContext()->name << ".";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
void FEMCapacity<TDataTypes, TSpatialTypes, TElementType>::addForce(
    const core::MechanicalParams* mparams,
    sofa::DataVecDeriv_t<DataTypes>& f,
    const sofa::DataVecCoord_t<DataTypes>& x,
    const sofa::DataVecDeriv_t<DataTypes>& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(f);
    SOFA_UNUSED(x);
    SOFA_UNUSED(v);
}

template <class TDataTypes, class TSpatialTypes, class TElementType>
SReal FEMCapacity<TDataTypes, TSpatialTypes, TElementType>::getPotentialEnergy(
    const core::MechanicalParams* mparams,
    const DataVecCoord_t<DataTypes>& x) const
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);
    return 0_sreal;
}

}  // namespace sofa::component::mass
