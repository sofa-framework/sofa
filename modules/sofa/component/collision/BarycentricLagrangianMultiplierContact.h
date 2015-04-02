/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_H

#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <sofa/component/constraintset/LagrangianMultiplierContactConstraint.h>
#include <sofa/helper/Factory.h>



namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template < class TCollisionModel1, class TCollisionModel2 >
class BarycentricLagrangianMultiplierContact : public core::collision::Contact
{
public:
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef core::collision::DetectionOutputVector OutputVector;
    typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;
    typedef typename CollisionModel1::DataTypes DataTypes1;
    typedef typename CollisionModel2::DataTypes DataTypes2;
    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    ContactMapper<CollisionModel1,DataTypes1> mapper1;
    ContactMapper<CollisionModel2,DataTypes2> mapper2;

    constraintset::LagrangianMultiplierContactConstraint<Vec3Types>* ff;
    core::objectmodel::BaseContext* parent;

    void setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2);

public:
    BarycentricLagrangianMultiplierContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    ~BarycentricLagrangianMultiplierContact();

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(OutputVector* outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    void removeResponse();

    void draw(const core::visual::VisualParams* vparams);
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
