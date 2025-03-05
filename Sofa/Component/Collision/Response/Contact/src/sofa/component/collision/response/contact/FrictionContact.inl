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
#include <sofa/component/collision/response/contact/FrictionContact.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/response/contact/CollisionResponse.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/IdentityContactMapper.h>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <sofa/simulation/Node.h>
#include <iostream>

namespace sofa::component::collision::response::contact
{

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::FrictionContact()
    : FrictionContact(nullptr, nullptr, nullptr)
{
}


template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::FrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, constraint::lagrangian::model::UnilateralLagrangianContactParameters, ResponseDataTypes>(model1,model2,intersectionMethod)
      , d_mu (initData(&d_mu, 0.8_sreal, "mu", "Friction coefficient (0 for frictionless contacts)"))
{

}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
constraint::lagrangian::model::UnilateralLagrangianContactParameters FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::getParameterFromDatas() const
{
    return {d_mu.getValue()};
}


template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes  >
void FrictionContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setupConstraint(MechanicalState1 * mmodel1,MechanicalState2 * mmodel2)
{
    this->m_constraint = sofa::core::objectmodel::New<constraint::lagrangian::model::UnilateralLagrangianConstraint<defaulttype::Vec3Types> >(mmodel1, mmodel2);
}


} //namespace sofa::component::collision::response::contact
