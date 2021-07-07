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

#include "EnslavementForceFeedback.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/type/Vec.h>

namespace sofa
{

namespace component
{

namespace controller
{
    EnslavementForceFeedback::EnslavementForceFeedback( core::CollisionModel* collModel1, core::CollisionModel* collModel2 )
    : ContactListener( collModel1, collModel2)
    , d_relativeStiffness(initData(&d_relativeStiffness, 4.0, "d_relativeStiffness", "Relative Stiffness"))
    , d_attractionDistance(initData(&d_attractionDistance, 0.3, "d_attractionDistance", "Distance at which the Omni is attracted to the contact point."))
    , d_normalsPointOut(initData(&d_normalsPointOut, true, "d_normalsPointOut", "True if the normals of objects point outwards, false if they point inwards."))
    , d_contactScale(initData(&d_contactScale, 1.0, "d_contactScale", "Scales the maximum penetration depth."))
    , d_penOffset(initData(&d_penOffset, 0.0, "penetrationOffset", "Distance at which there is no reaction force."))
    {
    }

    void EnslavementForceFeedback::init()
    {
        Inherit1::init();
        Inherit2::init();
    }

    void EnslavementForceFeedback::beginContact(const type::vector<const type::vector<core::collision::DetectionOutput>* >& contacts)
    {

        ContactListener::ContactVectorsIterator vecIter;
        ContactListener::ContactVectorsIterator lastVecIter = contacts.end();

        for(vecIter = contacts.begin(); vecIter != lastVecIter; ++vecIter)
        {
            ContactListener::ContactsIterator iter;
            ContactListener::ContactsIterator lastIter = (*vecIter)->end();

            for(iter = (*vecIter)->begin(); iter != lastIter; ++iter)
            {
                core::collision::DetectionOutput detectionOutput = (*iter);
                sofa::type::Vec3d model1Coord = detectionOutput.point[0];
                sofa::type::Vec3d model2Coord = detectionOutput.point[1];

                sofa::type::Vec3d u = model2Coord - model1Coord;
                sofa::type::Vec3d norm = detectionOutput.normal;
                double pen = u*norm;

                pen = (pen - d_penOffset.getValue()) / d_contactScale.getValue();

                if(!d_normalsPointOut.getValue())
                {
                    if(pen < 0 && pen > -1 * d_attractionDistance.getValue())
                    {
                        m_contactForce = -(norm * pen);
                    }
                    else if (pen >=0)
                    {
                        m_contactForce = (norm * pen * -1 * d_relativeStiffness.getValue());
                    }
                }
                else
                {
                    if( pen > 0 && pen < d_attractionDistance.getValue())
                    {
                        m_contactForce = -(norm * pen);
                    }
                    else if(pen <= 0)
                    {
                        m_contactForce = (norm * pen *-1 * d_relativeStiffness.getValue());
                    }
                }
            }
        }
    }

    void EnslavementForceFeedback::endContact(void*)
    {
        m_contactForce.set(0,0,0);
    }

    void EnslavementForceFeedback::computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz)
    {
        SOFA_UNUSED(x); SOFA_UNUSED(y); SOFA_UNUSED(z);
        SOFA_UNUSED(u); SOFA_UNUSED(v); SOFA_UNUSED(w); SOFA_UNUSED(q);

        fx = m_contactForce[0];
        fy = m_contactForce[1];
        fz = m_contactForce[2];
    }

    void EnslavementForceFeedback::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool, const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &V_tool_world, sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world )
    {
        SOFA_UNUSED(world_H_tool);
        SOFA_UNUSED(V_tool_world);
        SOFA_UNUSED(W_tool_world);
    }


int EnslavementForceFeedbackClass = core::RegisterObject("Updates force to the haptics device based on collisions")
        .add< EnslavementForceFeedback >()
        ;

}

}

}
