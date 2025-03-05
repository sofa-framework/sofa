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
#include <sofa/component/constraint/lagrangian/model/BaseContactLagrangianConstraint.h>
#include <sofa/component/collision/response/mapper/BaseContactMapper.h>
#include <sofa/component/collision/response/contact/ContactIdentifier.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::collision::response::contact
{
template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
class BaseUnilateralContactResponse : public core::collision::Contact, public ContactIdentifier
{
public:
    SOFA_CLASS(SOFA_TEMPLATE4(BaseUnilateralContactResponse, TCollisionModel1, TCollisionModel2,ConstraintParameters, ResponseDataTypes), core::collision::Contact);
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef typename TCollisionModel1::DataTypes::CPos TVec1;
    typedef typename TCollisionModel1::DataTypes::CPos TVec2;
    typedef sofa::defaulttype::StdVectorTypes<TVec1, TVec2, typename TCollisionModel1::DataTypes::Real > DataTypes1;
    typedef sofa::defaulttype::StdVectorTypes<TVec1, TVec2, typename TCollisionModel1::DataTypes::Real > DataTypes2;

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
    bool selfCollision; ///< true if model1==model2 (in this case, only mapper1 is used)
    mapper::ContactMapper<CollisionModel1,DataTypes1> mapper1;
    mapper::ContactMapper<CollisionModel2,DataTypes2> mapper2;

    typename constraint::lagrangian::model::BaseContactLagrangianConstraint<sofa::defaulttype::Vec3Types,ConstraintParameters>::SPtr m_constraint;
    core::objectmodel::BaseContext* parent;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_COLLISION_RESPONSE_CONTACT()
    sofa::core::objectmodel::lifecycle::RenamedData<double> mu;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_COLLISION_RESPONSE_CONTACT()
    sofa::core::objectmodel::lifecycle::RenamedData<double> tol;

    Data<double> d_tol; ///< tolerance for the constraints resolution (0 for default tolerance)
    std::vector< sofa::core::collision::DetectionOutput* > contacts;
    std::vector< std::pair< std::pair<int, int>, double > > mappedContacts;

    virtual void activateMappers();

    void setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2);

    BaseUnilateralContactResponse();
    BaseUnilateralContactResponse(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);

    ~BaseUnilateralContactResponse() override;
public:
    void cleanup() override;

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() override { return std::make_pair(model1,model2); }

    void setDetectionOutputs(OutputVector* outputs) override;

    void createResponse(core::objectmodel::BaseContext* group) override;

    void removeResponse() override;

    virtual ConstraintParameters getParameterFromDatas() const = 0;
    virtual void setupConstraint(MechanicalState1 *,MechanicalState2 *) = 0;
};

} // namespace sofa::component::collision::response::contact
