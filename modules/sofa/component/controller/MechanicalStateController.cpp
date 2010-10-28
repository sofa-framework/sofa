/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
//
// C++ Implementation : MechanicalStateController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_CPP
#include <sofa/component/controller/MechanicalStateController.inl>

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

SOFA_DECL_CLASS(MechanicalStateController)

// Register in the Factory
int MechanicalStateControllerClass = core::RegisterObject("Provides a Mouse & Keyboard user control on a Mechanical State.")
#ifndef SOFA_FLOAT
//.add< MechanicalStateController<Vec3dTypes> >()
//.add< MechanicalStateController<Vec2dTypes> >()
        .add< MechanicalStateController<Vec1dTypes> >()
        .add< MechanicalStateController<Rigid3dTypes> >()
//.add< MechanicalStateController<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
//.add< MechanicalStateController<Vec3fTypes> >()
//.add< MechanicalStateController<Vec2fTypes> >()
        .add< MechanicalStateController<Vec1fTypes> >()
        .add< MechanicalStateController<Rigid3fTypes> >()
//.add< MechanicalStateController<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec3dTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec2dTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec1dTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid3dTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec3fTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec2fTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec1fTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid3fTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid2fTypes>;
#endif

#ifndef SOFA_FLOAT
template <>
void MechanicalStateController<Vec1dTypes>::applyController()
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;


    //std::cout<<" applyController() : omni "<< omni << "  buttonOmni " <<buttonOmni<<std::endl;

    if(omni)
    {
        if(mState)
        {
            helper::WriteAccessor<Data<VecCoord> > x0 = *mState->write(sofa::core::VecCoordId::restPosition());

            if(buttonOmni)
            {
                if (x0[0].x() < -0.001)
                    x0[0].x() += 0.05;
                else
                    x0[0].x() =  -0.001;

                if (x0[1].x() > 0.001)
                    x0[1].x() -= 0.05;
                else
                    x0[1].x() = 0.001;
            }
            else
            {
                //sout<<"mouseMode==Release"<<sendl;

                if (x0[0].x() > -0.3)
                    x0[0].x() -= 0.05;
                else
                    x0[0].x() = -0.3;

                if (x0[1].x() < 0.3)
                    x0[1].x() += 0.05;
                else
                    x0[1].x() = 0.3;

            }


        }

    }
    else
    {
        //if (mState)
        //{
        //  helper::WriteAccessor<Data<VecCoord> > xfree = *mState->write(sofa::core::VecCoordId::restPosition());
        //	if (mouseMode==BtLeft || mouseMode==BtRight)
        //	{
        //			//sout<<"mouseMode==BtLeft"<<sendl;

        //			if (x0[0].x() < -0.01)
        //				x0[0].x() += 0.01;
        //			else
        //				x0[0].x() =  -0.01;
        //
        //			if (x0[1].x() > 0.01)
        //				x0[1].x() -= 0.01;
        //			else
        //				x0[1].x() = 0.01;

        //	}
        //	else
        //	{
        //			//sout<<"mouseMode==Release"<<sendl;

        //			if (x0[0].x() > -0.7)
        //				x0[0].x() -= 0.01;
        //			else
        //				x0[0].x() = -0.7;
        //
        //			if (x0[1].x() < 0.7)
        //				x0[1].x() += 0.01;
        //			else
        //				x0[1].x() = 0.7;

        //	}
        //}
    }



    //	//sofa::simulation::Node *node = static_cast<sofa::simulation::Node*> (this->getContext());
    //	//sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor; mechaVisitor.execute(node);
    //	//sofa::simulation::UpdateMappingVisitor updateVisitor; updateVisitor.execute(node);
    //}
};
#endif


#ifndef SOFA_DOUBLE
template <>
void MechanicalStateController<Vec1fTypes>::applyController()
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;


    //sout<<" applyController() : omni "<< omni << "  buttonOmni " <<buttonOmni<<sendl;

    if(omni)
    {
        if(mState)
        {
            helper::WriteAccessor<Data<VecCoord> > x0 = *mState->write(sofa::core::VecCoordId::restPosition());

            if(buttonOmni)
            {
                if (x0[0].x() < -0.001f)
                    x0[0].x() += 0.05f;
                else
                    x0[0].x() =  -0.001f;

                if (x0[1].x() > 0.001f)
                    x0[1].x() -= 0.05f;
                else
                    x0[1].x() = 0.001f;
            }
            else
            {
                //sout<<"mouseMode==Release"<<sendl;

                if (x0[0].x() > -0.7f)
                    x0[0].x() -= 0.05f;
                else
                    x0[0].x() = -0.7f;

                if (x0[1].x() < 0.7f)
                    x0[1].x() += 0.05f;
                else
                    x0[1].x() = 0.7f;

            }


        }

    }
    else
    {
        //if (mState)
        //{
        //	if (mouseMode==BtLeft || mouseMode==BtRight)
        //	{
        //			//sout<<"mouseMode==BtLeft"<<sendl;

        //			if (x0[0].x() < -0.01f)
        //				x0[0].x() += 0.01f;
        //			else
        //				x0[0].x() =  -0.01f;
        //
        //			if (x0[1].x() > 0.01f)
        //				x0[1].x() -= 0.01f;
        //			else
        //				x0[1].x() = 0.01f;

        //	}
        //	else
        //	{
        //			//sout<<"mouseMode==Release"<<sendl;

        //			if (x0[0].x() > -0.7f)
        //				x0[0].x() -= 0.01f;
        //			else
        //				x0[0].x() = -0.7f;
        //
        //			if (x0[1].x() < 0.7f)
        //				x0[1].x() += 0.01f;
        //			else
        //				x0[1].x() = 0.7f;

        //	}
        //}
    }



    //	//sofa::simulation::Node *node = static_cast<sofa::simulation::Node*> (this->getContext());
    //	//sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor; mechaVisitor.execute(node);
    //	//sofa::simulation::UpdateMappingVisitor updateVisitor; updateVisitor.execute(node);
    //}
};
#endif

#ifndef SOFA_DOUBLE
template <>
void MechanicalStateController<Vec1fTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
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
        applyController();

}
#endif

#ifndef SOFA_FLOAT
template <>
void MechanicalStateController<Vec1dTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
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
        applyController();

}
#endif

#ifndef SOFA_FLOAT
template <>
void MechanicalStateController<Rigid3dTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
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
            applyController();
        break;

    default :
        break;
    }
}
#endif

#ifndef SOFA_DOUBLE
template <>
void MechanicalStateController<Rigid3fTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
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
            applyController();
        break;

    default :
        break;
    }
}
#endif

} // namespace controller

} // namespace component

} // namespace sofa
