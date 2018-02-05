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
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICSTICKCONTACT_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICSTICKCONTACT_H
#include "config.h"

#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <SofaGeneralDeformable/VectorSpringForceField.h>


namespace sofa
{

namespace component
{

namespace collision
{


template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
class BarycentricStickContact : public core::collision::Contact
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(BarycentricStickContact,TCollisionModel1,TCollisionModel2,ResponseDataTypes), core::collision::Contact);
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
    typedef interactionforcefield::VectorSpringForceField<ResponseDataTypes> ResponseForceField;

protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    ContactMapper<CollisionModel1,DataTypes1> mapper1;
    ContactMapper<CollisionModel2,DataTypes2> mapper2;

    typename ResponseForceField::SPtr ff;
    core::objectmodel::BaseContext* parent;

    typedef std::map<core::collision::DetectionOutput::ContactId,int> ContactIndexMap;
    /// Mapping of contactids to force element (+1, so that 0 means not active).
    /// This allows to ignore duplicate contacts, and preserve information associated with each contact point over time
    ContactIndexMap contactIndex;

    BarycentricStickContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    ~BarycentricStickContact();

    void setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2);

public:
    Data<bool> f_keepAlive;

    void cleanup() override;

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() override { return std::make_pair(model1,model2); }

    void setDetectionOutputs(OutputVector* outputs) override;

    void createResponse(core::objectmodel::BaseContext* group) override;

    void removeResponse() override;

    /// Return true if this contact should be kept alive, even if objects are no longer in collision
    virtual bool keepAlive() override { return f_keepAlive.getValue(); }

    /// Control the keepAlive flag of the contact.
    virtual void setKeepAlive(bool val) override { f_keepAlive.setValue(val); }

    void draw(const core::visual::VisualParams* vparams) override;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
