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
#include <sofa/component/collision/response/contact/PenalityContactForceField.h>
#include <sofa/helper/Factory.h>

#include <sofa/component/collision/response/mapper/RigidContactMapper.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/IdentityContactMapper.h>

#include <sofa/component/collision/geometry/CylinderModel.h>


namespace sofa::component::collision::response::contact
{

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
class BarycentricPenalityContact : public core::collision::Contact
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(BarycentricPenalityContact, TCollisionModel1, TCollisionModel2, ResponseDataTypes), core::collision::Contact);

    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef core::collision::DetectionOutputVector OutputVector;
    typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;
    typedef ResponseDataTypes DataTypes1;
    typedef ResponseDataTypes DataTypes2;
    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
    typedef PenalityContactForceField<ResponseDataTypes> ResponseForceField;
protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    mapper::ContactMapper<CollisionModel1,DataTypes1> mapper1;
    mapper::ContactMapper<CollisionModel2,DataTypes2> mapper2;

    typename ResponseForceField::SPtr ff;
    core::objectmodel::BaseContext* parent;

    typedef std::map<core::collision::DetectionOutput::ContactId, int> ContactIndexMap;
    /// Mapping of contactids to force element (+1, so that 0 means not active).
    /// This allows to ignore duplicate contacts, and preserve information associated with each contact point over time
    ContactIndexMap contactIndex;

    BarycentricPenalityContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    ~BarycentricPenalityContact() override;

    void setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2);

public:
    void cleanup() override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    std::pair<core::CollisionModel*,core::CollisionModel*> doGetCollisionModels() override 
    { 
        return std::make_pair(model1,model2); 
    }

    void doSetDetectionOutputs(OutputVector* outputs) override;

    void doCreateResponse(core::objectmodel::BaseContext* group) override;

    void doRemoveResponse() override;

};

#if !defined(SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_CPP)
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::RigidSphereModel>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::RigidSphereModel, collision::geometry::RigidSphereModel>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::RigidSphereModel, collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::RigidSphereModel>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::RigidSphereModel>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;

extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
extern template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricPenalityContact<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, collision::geometry::RigidSphereModel>;
#endif

} //namespace sofa::component::collision::response::contact
