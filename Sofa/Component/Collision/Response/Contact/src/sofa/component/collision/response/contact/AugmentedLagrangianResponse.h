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
#include <sofa/component/collision/response/contact/config.h>

#include <sofa/helper/Factory.h>
#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/constraint/lagrangian/model/AugmentedLagrangianConstraint.h>
#include <sofa/component/collision/response/contact/BaseUnilateralContactResponse.h>
#include <sofa/component/collision/response/contact/ContactIdentifier.h>

#include <sofa/core/objectmodel/RenamedData.h>

namespace sofa::component::collision::response::contact
{
template <class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
class AugmentedLagrangianResponse : public BaseUnilateralContactResponse<TCollisionModel1,  TCollisionModel2,constraint::lagrangian::model::AugmentedLagrangianContactParameters, ResponseDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(AugmentedLagrangianResponse, TCollisionModel1, TCollisionModel2, ResponseDataTypes), SOFA_TEMPLATE4(BaseUnilateralContactResponse, TCollisionModel1, TCollisionModel2,constraint::lagrangian::model::AugmentedLagrangianContactParameters, ResponseDataTypes));

    typedef typename Inherit1::DataTypes1 DataTypes1;
    typedef typename Inherit1::DataTypes2 DataTypes2;
    typedef typename Inherit1::CollisionModel1 CollisionModel1;
    typedef typename Inherit1::CollisionModel2 CollisionModel2;
    typedef typename Inherit1::Intersection Intersection;

    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;

    Data<double> d_mu; ///< friction parameter
    Data<double> d_epsilon; ///< Penalty parameter

    AugmentedLagrangianResponse();
    AugmentedLagrangianResponse(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);

    virtual ~AugmentedLagrangianResponse() = default;

    virtual constraint::lagrangian::model::AugmentedLagrangianContactParameters getParameterFromDatas() const override;
    virtual void setupConstraint(MechanicalState1 *,MechanicalState2 *) override;

};

} // namespace sofa::component::collision::response::contact
