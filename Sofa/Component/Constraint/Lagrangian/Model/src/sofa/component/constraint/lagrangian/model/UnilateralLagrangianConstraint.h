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
#include <sofa/component/constraint/lagrangian/model/config.h>

#include <sofa/core/behavior/PairInteractionConstraint.h>
#include <sofa/core/behavior/ConstraintResolution.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/constraint/lagrangian/model/BaseContactLagrangianConstraint.h>
#include <sofa/component/constraint/lagrangian/model/UnilateralConstraintResolution.h>
#include <iostream>
#include <map>
#include <deque>

namespace sofa::component::constraint::lagrangian::model
{

struct UnilateralLagrangianContactParameters final : public BaseContactParams
{
    UnilateralLagrangianContactParameters() : mu(0.0) {};
    UnilateralLagrangianContactParameters(SReal _mu) : mu(_mu) {};

    virtual bool hasTangentialComponent() const override
    {
        return mu>0.0;
    }

    SReal mu;
};

template<class DataTypes>
class UnilateralLagrangianConstraint : public BaseContactLagrangianConstraint<DataTypes,UnilateralLagrangianContactParameters>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UnilateralLagrangianConstraint,DataTypes), SOFA_TEMPLATE2(BaseContactLagrangianConstraint,DataTypes,UnilateralLagrangianContactParameters));
    typedef BaseContactLagrangianConstraint<DataTypes,UnilateralLagrangianContactParameters> Inherit;
    typedef typename Inherit::MechanicalState MechanicalState;
    typedef typename Inherit::Contact Contact;

protected:
    UnilateralLagrangianConstraint(MechanicalState* object1=nullptr, MechanicalState* object2=nullptr);
    virtual ~UnilateralLagrangianConstraint() = default;

public:
    virtual void getConstraintResolution(const core::ConstraintParams *,std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset) override;

};


#if !defined(SOFA_COMPONENT_CONSTRAINTSET_UNILATERALLAGRANGIANCONSTRAINT_CPP)
    extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API BaseContactLagrangianConstraint<defaulttype::Vec3Types,UnilateralLagrangianContactParameters>;
    extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API UnilateralLagrangianConstraint<defaulttype::Vec3Types>;
#endif


} //namespace sofa::component::constraint::lagrangian::model
