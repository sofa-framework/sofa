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

#include <sofa/component/mass/config.h>
#include <sofa/core/BaseNodalProperty.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::component::mass
{

template<class Scalar>
class NodalMassDensity : public sofa::core::BaseNodalProperty<Scalar>
{
public:
    SOFA_CLASS(NodalMassDensity<Scalar>, sofa::core::BaseNodalProperty<Scalar>);

    template<class T>
    static bool canCreate(T* obj, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        if (const auto* state = context->getMechanicalState())
        {
            static const auto scalarType = defaulttype::DataTypeInfo<Scalar>::name();
            if (state->getScalarType() == scalarType)
                return true;
            arg->logError("The mechanical state does not have a scalar type of '" + scalarType + "'");
            return false;
        }
        return true;
    }

private:

    static constexpr Scalar defaultMassDensity = static_cast<Scalar>(1.);

    NodalMassDensity() : sofa::core::BaseNodalProperty<Scalar>(defaultMassDensity) {}
};

}
