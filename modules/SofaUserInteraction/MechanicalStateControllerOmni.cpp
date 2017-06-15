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
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLEROMNI_CPP
#include <SofaUserInteraction/MechanicalStateControllerOmni.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MechanicalStateControllerOmni)

// Register in the Factory
int MechanicalStateControllerOmniClass = core::RegisterObject("Provides an Omni user control on a Mechanical State.")
#ifndef SOFA_FLOAT
//.add< MechanicalStateControllerOmni<Vec3dTypes> >()
//.add< MechanicalStateControllerOmni<Vec2dTypes> >()
        .add< MechanicalStateControllerOmni<Vec1dTypes> >()
        .add< MechanicalStateControllerOmni<Rigid3dTypes> >()
//.add< MechanicalStateControllerOmni<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
//.add< MechanicalStateControllerOmni<Vec3fTypes> >()
//.add< MechanicalStateControllerOmni<Vec2fTypes> >()
        .add< MechanicalStateControllerOmni<Vec1fTypes> >()
        .add< MechanicalStateControllerOmni<Rigid3fTypes> >()
//.add< MechanicalStateControllerOmni<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
//template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Vec3dTypes>;
//template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Vec2dTypes>;
template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Vec1dTypes>;
template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Rigid3dTypes>;
//template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
//template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Vec3fTypes>;
//template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Vec2fTypes>;
template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Vec1fTypes>;
template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Rigid3fTypes>;
//template class SOFA_USER_INTERACTION_API MechanicalStateControllerOmni<Rigid2fTypes>;
#endif

#ifndef SOFA_FLOAT
template <>
void MechanicalStateControllerOmni<Vec1dTypes>::applyController(const double dt)
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;

    if(device)
    {
        if(mState)
        {
            helper::WriteAccessor<Data<VecCoord> > x0 = *mState->write(sofa::core::VecCoordId::restPosition());
            const Real maxAngle = this->angle.getValue() * (Real)(M_PI/180.0);
            const Real speed = this->speed.getValue() * (Real)(M_PI/180.0);
            if(buttonDeviceState.getValue())
            {
                double angle = x0[0].x() - dt*speed; if (angle<0) angle = 0;
                x0[0].x() = angle;
                x0[1].x() = angle;
            }
            else
            {
                double angle = x0[0].x() + dt*speed; if (angle>maxAngle) angle = maxAngle;
                x0[0].x() = angle;
                x0[1].x() = angle;
            }
        }
    }

};
#endif


#ifndef SOFA_DOUBLE
template <>
void MechanicalStateControllerOmni<Vec1fTypes>::applyController(const double dt)
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;

    if(device)
    {
        if(mState)
        {
            helper::WriteAccessor<Data<VecCoord> > x0 = *mState->write(sofa::core::VecCoordId::restPosition());
            if(buttonDeviceState.getValue())
            {
                float angle = x0[0].x() - (float)dt; if (angle<0.05f) angle = 0.05f;
                x0[0].x() = angle;
                x0[1].x() = angle;
            }
            else
            {
                float angle = x0[0].x() + (float)dt; if (angle>0.5f) angle = 0.5f;
                x0[0].x() = angle;
                x0[1].x() = angle;
            }
        }
    }

};
#endif
/*
#ifndef SOFA_DOUBLE
template <>
void MechanicalStateControllerOmni<Vec1fTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
	//sout<<"MouseEvent detected"<<sendl;
	eventX = mev->getPosX();
	eventY = mev->getPosY();

	switch (mev->getState())
	{
		case sofa::core::objectmodel::MouseEvent::LeftPressed :
			mouseMode = BtLeft;
			break;

		case sofa::core::objectmodel::MouseEvent::LeftReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::RightPressed :
			mouseMode = BtRight;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::RightReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddlePressed :
			mouseMode = BtMiddle;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddleReleased :
			mouseMode = None;
			break;

		default :
			break;
	}
	if (handleEventTriggersUpdate.getValue())
        applyController(0);

}
#endif

#ifndef SOFA_FLOAT
template <>
void MechanicalStateControllerOmni<Vec1dTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
	//sout<<"MouseEvent detected"<<sendl;
	eventX = mev->getPosX();
	eventY = mev->getPosY();

	switch (mev->getState())
	{
		case sofa::core::objectmodel::MouseEvent::LeftPressed :
			mouseMode = BtLeft;
			break;

		case sofa::core::objectmodel::MouseEvent::LeftReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::RightPressed :
			mouseMode = BtRight;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::RightReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddlePressed :
			mouseMode = BtMiddle;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddleReleased :
			mouseMode = None;
			break;

		default :
			break;
	}
	if (handleEventTriggersUpdate.getValue())
        applyController(0);

}
#endif

#ifndef SOFA_FLOAT
template <>
void MechanicalStateControllerOmni<Rigid3dTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
	//sout<<"MouseEvent detected"<<sendl;
	eventX = mev->getPosX();
	eventY = mev->getPosY();

	switch (mev->getState())
	{
		case sofa::core::objectmodel::MouseEvent::LeftPressed :
			mouseMode = BtLeft;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::LeftReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::RightPressed :
			mouseMode = BtRight;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::RightReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddlePressed :
			mouseMode = BtMiddle;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddleReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::Move :
			if (handleEventTriggersUpdate.getValue())
                applyController(0);
			break;

		default :
			break;
	}
}
#endif

#ifndef SOFA_DOUBLE
template <>
void MechanicalStateControllerOmni<Rigid3fTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
	//sout<<"MouseEvent detected"<<sendl;
	eventX = mev->getPosX();
	eventY = mev->getPosY();

	switch (mev->getState())
	{
		case sofa::core::objectmodel::MouseEvent::LeftPressed :
			mouseMode = BtLeft;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::LeftReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::RightPressed :
			mouseMode = BtRight;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::RightReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddlePressed :
			mouseMode = BtMiddle;
			mouseSavedPosX = eventX;
			mouseSavedPosY = eventY;
			break;

		case sofa::core::objectmodel::MouseEvent::MiddleReleased :
			mouseMode = None;
			break;

		case sofa::core::objectmodel::MouseEvent::Move :
			if (handleEventTriggersUpdate.getValue())
                applyController(0);
			break;

		default :
			break;
	}
}
#endif
*/
} // namespace controller

} // namespace component

} // namespace sofa
