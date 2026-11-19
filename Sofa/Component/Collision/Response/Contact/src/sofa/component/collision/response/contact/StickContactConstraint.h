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

#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/component/constraint/lagrangian/model/BilateralLagrangianConstraint.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/collision/response/mapper/BaseContactMapper.h>
#include <sofa/component/collision/response/contact/ContactIdentifier.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/BaseMapping.h>

namespace sofa::component::collision::response::contact
{

template <class TCollisionModel1, class TCollisionModel2>
class StickContactConstraint : public core::collision::Contact, public ContactIdentifier
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(StickContactConstraint, TCollisionModel1, TCollisionModel2), core::collision::Contact);
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef typename CollisionModel1::DataTypes DataTypes1;
    typedef typename CollisionModel2::DataTypes DataTypes2;
    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
    typedef core::collision::DetectionOutputVector OutputVector;
    typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;

protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;
    mapper::ContactMapper<CollisionModel1,DataTypes1> mapper1;
    mapper::ContactMapper<CollisionModel2,DataTypes2> mapper2;

    constraint::lagrangian::model::BilateralLagrangianConstraint<sofa::defaulttype::Vec3Types>::SPtr m_constraint;
    core::objectmodel::BaseContext* parent;

    std::vector< sofa::core::collision::DetectionOutput* > contacts;
    std::vector< std::pair< std::pair<int, int>, double > > mappedContacts;
    void activateMappers();

    StickContactConstraint();
    StickContactConstraint(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);

    ~StickContactConstraint() override;

public:
    Data<bool> d_keepAlive; ///< set to true to keep this contact alive even after collisions are no longer detected

    /// Return true if this contact should be kept alive, even if objects are no longer in collision
    bool keepAlive() override { return d_keepAlive.getValue(); }

    /// Control the keepAlive flag of the contact.
    void setKeepAlive(bool val) override { d_keepAlive.setValue(val); }

    void cleanup() override;

protected:
    std::pair<core::CollisionModel*,core::CollisionModel*> doGetCollisionModels() override 
    { 
        return std::make_pair(model1,model2); 
    }

    void doSetDetectionOutputs(OutputVector* outputs) override;

    void doCreateResponse(core::objectmodel::BaseContext* group) override;

    void doRemoveResponse() override;
};


} //namespace sofa::component::collision::response::contact
