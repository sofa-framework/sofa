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

#include <sofa/core/config.h>

#include <sofa/core/objectmodel/BaseContext.h>

namespace sofa::core
{

struct SOFA_CORE_API BaseTemplateDeductionRule
{
    bool doesComponentComplyWith(
        objectmodel::BaseContext* context,
        objectmodel::BaseObjectDescription* arg)
    {
        if (!context) return false;
        return doDoesComponentComplyWith(context, arg);
    };

protected:

    virtual bool doDoesComponentComplyWith(
        objectmodel::BaseContext* context,
        objectmodel::BaseObjectDescription* arg) = 0;
};

template<class... T>
struct OtherComponentsInContextDeductionRule : public BaseTemplateDeductionRule
{
protected:
    bool doDoesComponentComplyWith(
        objectmodel::BaseContext* context,
        objectmodel::BaseObjectDescription* arg) override
    {
        SOFA_UNUSED(arg);
        return ((context->get<T>() != nullptr) && ...);
    }
};

template<class DataTypes>
using MechanicalStateDeductionRule = OtherComponentsInContextDeductionRule<sofa::core::behavior::MechanicalState<DataTypes>>;

template<class T>
struct CanCreateDeductionRule : public BaseTemplateDeductionRule
{
protected:
    bool doDoesComponentComplyWith(
        objectmodel::BaseContext* context,
        objectmodel::BaseObjectDescription* arg) override
    {
        T* instance = nullptr;
        return T::canCreate(instance, context, arg);
    }
};


}
