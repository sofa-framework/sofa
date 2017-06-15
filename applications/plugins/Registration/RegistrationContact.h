/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_REGISTRATIONCONTACT_H
#define SOFA_COMPONENT_COLLISION_REGISTRATIONCONTACT_H

#include "RegistrationContactForceField.h"
#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
//#include <SofaBaseMechanics/BarycentricMapping.h>
#include <sofa/helper/Factory.h>
#include <SofaBaseCollision/BaseContactMapper.h>

#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMeshCollision/IdentityContactMapper.h>
#include <SofaMeshCollision/RigidContactMapper.inl>


namespace sofa
{

  namespace component
  {

    namespace collision
    {


    template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes = sofa::defaulttype::Vec3Types >
	class RegistrationContact : public core::collision::Contact
	{
	public:
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
	typedef interactionforcefield::RegistrationContactForceField<ResponseDataTypes> ResponseForceField;

	protected:
	CollisionModel1* model1;
	CollisionModel2* model2;
	Intersection* intersectionMethod;

	ContactMapper<CollisionModel1,DataTypes1> mapper1;
	ContactMapper<CollisionModel2,DataTypes2> mapper2;

	ResponseForceField* ff;
	core::objectmodel::BaseContext* parent;
	
	typedef std::map<core::collision::DetectionOutput::ContactId,int> ContactIndexMap;
	/// Mapping of contactids to force element (+1, so that 0 means not active).
	/// This allows to ignore duplicate contacts, and preserve information associated with each contact point over time
	ContactIndexMap contactIndex;
	public:
	RegistrationContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
	~RegistrationContact();
	
	void cleanup();
	
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
