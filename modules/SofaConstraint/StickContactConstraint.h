/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_STICKCONTACTCONSTRAINT_H
#define SOFA_COMPONENT_COLLISION_STICKCONTACTCONSTRAINT_H
#include "config.h"

#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaConstraint/BilateralInteractionConstraint.h>
#include <sofa/helper/Factory.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaConstraint/ContactIdentifier.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/BaseMapping.h>

namespace sofa
{

namespace component
{

namespace collision
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
    ContactMapper<CollisionModel1,DataTypes1> mapper1;
    ContactMapper<CollisionModel2,DataTypes2> mapper2;

    constraintset::BilateralInteractionConstraint<sofa::defaulttype::Vec3Types>::SPtr m_constraint;
    core::objectmodel::BaseContext* parent;

    std::vector< sofa::core::collision::DetectionOutput* > contacts;
    std::vector< std::pair< std::pair<int, int>, double > > mappedContacts;
    void activateMappers();

    StickContactConstraint() : model1(NULL), model2(NULL), intersectionMethod(NULL), parent(NULL) {}

    StickContactConstraint(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    virtual ~StickContactConstraint();
public:
    Data<bool> f_keepAlive;

    /// Return true if this contact should be kept alive, even if objects are no longer in collision
    virtual bool keepAlive() override { return f_keepAlive.getValue(); }

    /// Control the keepAlive flag of the contact.
    virtual void setKeepAlive(bool val) override { f_keepAlive.setValue(val); }


    void cleanup() override;

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() override { return std::make_pair(model1,model2); }

    void setDetectionOutputs(OutputVector* outputs) override;

    void createResponse(core::objectmodel::BaseContext* group) override;

    void removeResponse() override;
};


} // collision

} // component

} // sofa

#endif // SOFA_COMPONENT_COLLISION_STICKCONTACTCONSTRAINT_H
