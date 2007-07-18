/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#ifdef SOFA_TEST_FRICTION
#include <sofa/component/forcefield/PenalityContactFrictionForceField.h>
#else
#include <sofa/component/forcefield/PenalityContactForceField.h>
#endif
#include <sofa/helper/Factory.h>
#include <sofa/component/collision/BarycentricContactMapper.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template < class TCollisionModel1, class TCollisionModel2 >
class BarycentricPenalityContact : public core::componentmodel::collision::Contact, public core::VisualModel
{
public:
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::componentmodel::collision::Intersection Intersection;
    typedef core::componentmodel::behavior::MechanicalState<typename CollisionModel1::DataTypes> MechanicalState1;
    typedef core::componentmodel::behavior::MechanicalState<typename CollisionModel2::DataTypes> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
#ifdef SOFA_TEST_FRICTION
    typedef forcefield::PenalityContactFrictionForceField<Vec3Types> ResponseForceField;
#else
    typedef forcefield::PenalityContactForceField<Vec3Types> ResponseForceField;
#endif
protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    BarycentricContactMapper<CollisionModel1> mapper1;
    BarycentricContactMapper<CollisionModel2> mapper2;

    ResponseForceField* ff;
    core::objectmodel::BaseContext* parent;

    typedef std::map<core::componentmodel::collision::DetectionOutput::ContactId,int> ContactIndexMap;
    /// Mapping of contactids to force element (+1, so that 0 means not active).
    /// This allows to ignore duplicate contacts, and preserve information associated with each contact point over time
    ContactIndexMap contactIndex;
public:
    BarycentricPenalityContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    ~BarycentricPenalityContact();

    void cleanup();

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(std::vector<core::componentmodel::collision::DetectionOutput>& outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    void removeResponse();

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
