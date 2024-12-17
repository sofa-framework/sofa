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
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_CPP
#include <sofa/component/controller/MechanicalStateController.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::controller
{

using namespace sofa::defaulttype;

template <>
void MechanicalStateController<Vec1Types>::applyController()
{
    using sofa::type::Vec;


    if(mState)
    {
        helper::WriteAccessor<Data<VecCoord> > x0 = *mState->write(sofa::core::vec_id::write_access::restPosition);
        if(buttonDevice)
        {
            if (x0[0].x() < -0.1) //angle de fermeture max
                x0[0].x() += 0.01; //vitesse de fermeture
            else
                x0[0].x() =  -0.1;
            /*
                            if (x0[1].x() > 0.001)
                                x0[1].x() -= 0.05;
                            else
                                x0[1].x() = 0.001;*/
        }
        else
        {
            if (x0[0].x() > -0.5)	 //angle d'ouverture max
                x0[0].x() -= 0.05;   //vitesse d'ouverture
            else
                x0[0].x() = -0.5;

            //if (x0[1].x() < 0.3)
            //	x0[1].x() += 0.05;
            //else
            //	x0[1].x() = 0.3;

        }


    }

//	}
    /*else
    {*/
    if (mState)
    {
        helper::WriteAccessor<Data<VecCoord> > x0 = *mState->write(sofa::core::vec_id::write_access::restPosition);
        if (mouseMode==BtMiddle)
        {
            x0[0].x() =  -0.4;
            x0[1].x() =  -0.4;

        }
        else
        {
            x0[0].x() =  0.0;
            x0[1].x() =  0.0;

        }
    }
}



template <>
void MechanicalStateController<Vec1Types>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
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
    if (d_handleEventTriggersUpdate.getValue())
        applyController();

}


template <>
void MechanicalStateController<Rigid3Types>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
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
        if (d_handleEventTriggersUpdate.getValue())
            applyController();
        break;

    default :
        break;
    }
}

void registerMechanicalStateController(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Provides a Mouse & Keyboard user control on a Mechanical State.")
        .add< MechanicalStateController<Vec1Types> >()
        .add< MechanicalStateController<Rigid3Types> >());
}

template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec1Types>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid3Types>;


} //namespace sofa::component::controller
