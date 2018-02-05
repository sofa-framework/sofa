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

#include "EnslavementForceFeedback.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace controller
{
	EnslavementForceFeedback::EnslavementForceFeedback( core::CollisionModel* collModel1, core::CollisionModel* collModel2 )
	: ContactListener( collModel1, collModel2)
    , relativeStiffness(initData(&relativeStiffness, 4.0, "relativeStiffness", "Relative Stiffness"))
    , attractionDistance(initData(&attractionDistance, 0.3, "attractionDistance", "Distance at which the Omni is attracted to the contact point."))
    , normalsPointOut(initData(&normalsPointOut, true, "normalsPointOut", "True if the normals of objects point outwards, false if they point inwards."))
    , contactScale(initData(&contactScale, 1.0, "contactScale", "Scales the maximum penetration depth."))
    , penOffset(initData(&penOffset, 0.0, "penetrationOffset", "Distance at which there is no reaction force."))
	{
	}

    void EnslavementForceFeedback::init()
    {
        this->ForceFeedback::init();
        this->ContactListener::init();
    }

	void EnslavementForceFeedback::beginContact(const helper::vector<const helper::vector<core::collision::DetectionOutput>* >& contacts)
	{
	
		ContactListener::ContactVectorsIterator vecIter;
		ContactListener::ContactVectorsIterator lastVecIter = contacts.end();

		//this->endContact(NULL);
		for(vecIter = contacts.begin(); vecIter != lastVecIter; ++vecIter)
		{
			ContactListener::ContactsIterator iter;
			ContactListener::ContactsIterator lastIter = (*vecIter)->end();

			for(iter = (*vecIter)->begin(); iter != lastIter; ++iter)
			{
				core::collision::DetectionOutput detectionOutput = (*iter);
				sofa::defaulttype::Vec3d model1Coord = detectionOutput.point[0];
				sofa::defaulttype::Vec3d model2Coord = detectionOutput.point[1];

				sofa::defaulttype::Vec3d u = model2Coord - model1Coord;
				sofa::defaulttype::Vec3d norm = detectionOutput.normal;
				double pen = u*norm; 

                pen = (pen - penOffset.getValue()) / contactScale.getValue();
                
                if(!normalsPointOut.getValue())
                {
                    if(pen < 0 && pen > -1 * attractionDistance.getValue())
                    {
                        contactForce = -(norm * pen);
                    }
                    else if (pen >=0)
                    {
                        contactForce = (norm * pen * -1 * relativeStiffness.getValue());
                    }
                }
                else
                {
                    if( pen > 0 && pen < attractionDistance.getValue())
                    {
                        contactForce = -(norm * pen);
                    }
                    else if(pen <= 0)
                    {
                        contactForce = (norm * pen *-1 * relativeStiffness.getValue());
                    }
                }
			}
		}
	}

	void EnslavementForceFeedback::endContact(void*)
	{
        contactForce = contactForce*0;
	}

    void EnslavementForceFeedback::computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz)
    {
            fx = contactForce[0];
            fy = contactForce[1];
            fz = contactForce[2];
    }

    void EnslavementForceFeedback::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool, const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &V_tool_world, sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world )
    {
    }


int EnslavementForceFeedbackClass = core::RegisterObject("Updates force to the haptics device based on collisions")
        .add< EnslavementForceFeedback >()
        ;

SOFA_DECL_CLASS(EnslavementForceFeedback)

}

}

}
