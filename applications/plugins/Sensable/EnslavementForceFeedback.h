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
#ifndef SOFA_COMPONENT_CONTROLLER_ENSLAVEMENTFORCEFEEDBACK_H
#define SOFA_COMPONENT_CONTROLLER_ENSLAVEMENTFORCEFEEDBACK_H

#include <SofaBaseCollision/ContactListener.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/core/collision/Intersection.h>
#include <SofaHaptics/ForceFeedback.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace controller
{
class EnslavementForceFeedback : public virtual core::collision::ContactListener, public sofa::component::controller::ForceFeedback
{
public:
    SOFA_CLASS2(EnslavementForceFeedback, core::collision::ContactListener, sofa::component::controller::ForceFeedback);

	EnslavementForceFeedback( core::CollisionModel* collModel1 = NULL, core::CollisionModel* collModel2 = NULL );

    ~EnslavementForceFeedback(){}

    virtual void init();
	virtual void beginContact(const helper::vector<const helper::vector<core::collision::DetectionOutput>* >& );
	virtual void endContact(void*);
    virtual void computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz);
    virtual void computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool, const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &V_tool_world, sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world );

protected:
	sofa::defaulttype::Vec3d contactForce;
    Data<double> relativeStiffness;
    Data<double> attractionDistance;
    Data<bool> normalsPointOut;
    Data<double> contactScale;
    Data<double> penOffset;


};

}
}
}

#endif //SOFA_COMPONENT_CONTROLLER_ENSLAVEMENTFORCEFEEDBACK_H
