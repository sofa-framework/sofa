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
#ifndef SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_INL
#define SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_INL

#include <ARTrackController.h>
#include <ARTrackEvent.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
void ARTrackController<DataTypes>::init()
{
}

template <>
void ARTrackController<Vec1dTypes>::init()
{
    getContext()->get<sofa::component::container::ArticulatedHierarchyContainer::ArticulationCenter::Articulation>(&articulations);
}

template <>
void ARTrackController<Vec3dTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    if(mstate)
    {
        if(!(*mstate->getXfree()).empty() && !(*mstate->getX()).empty())
        {
            for (unsigned int i=0; i<3; ++i)
            {
                std::cout<<"finger["<<i<<"] = "<<aev->getFingerposition(i)<<std::endl;
                (*mstate->getXfree())[i] = aev->getFingerposition(i);
                (*mstate->getX())[i] = aev->getFingerposition(i);
            }
        }
    }
}

template <>
void ARTrackController<RigidTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    if(mstate)
    {
        if(!(*mstate->getXfree()).empty() && !(*mstate->getX()).empty())
        {
            if(f_printLog.getValue())
                std::cout<<" aev pos :"<<aev->getPosition()<<" aev quat :"<<aev->getOrientation()<<std::endl;

            (*mstate->getXfree())[0].getCenter() = aev->getPosition();
            (*mstate->getX())[0].getCenter() = aev->getPosition();

            (*mstate->getXfree())[0].getOrientation() = aev->getOrientation();
            (*mstate->getX())[0].getOrientation() = aev->getOrientation();

            (*mstate->getXfree())[0].getOrientation().normalize();
            (*mstate->getX())[0].getOrientation().normalize();

        }
    }
}

template <>
void ARTrackController<Vec1dTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    if(mstate)
    {
        if(!(*mstate->getXfree()).empty() && !(*mstate->getX()).empty())
        {

            if(f_printLog.getValue())
                std::cout<<"pouce :"<<aev->getAngles()[0]<<"  index:"<<aev->getAngles()[1]<<"  autres:"<<aev->getAngles()[2]<<std::endl;


            for (unsigned int i=6; i<9; ++i) // thumb
            {
                (*mstate->getX0())[i].x() = (aev->getAngles()[0]);// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                /* if((*mstate->getX0())[i].x()<0)
                 {
                     (*mstate->getX0())[i] = 0.0;
                     (*mstate->getX())[i] = 0.0;
                 }*/
            }

            for (unsigned int i=9; i<12; ++i) // index
            {
                (*mstate->getX0())[i].x() = (aev->getAngles()[1]);// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                /* if((*mstate->getX0())[i].x()<0)
                 {
                     (*mstate->getX0())[i] = 0.0;
                     (*mstate->getX())[i] = 0.0;
                 }*/
            }

            for(unsigned int i=12; i<21; ++i) // middle, ring, little.
            {
                (*mstate->getX0())[i].x() = (aev->getAngles()[2]);// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                /*if((*mstate->getX0())[i].x()<0)
                {
                    (*mstate->getX0())[i] = 0.0;
                    (*mstate->getX())[i] = 0.0;
                }*/
            }
        }
    }
}

template <class DataTypes>
void ARTrackController<DataTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent* /*aev*/)
{
}

template <class DataTypes>
void ARTrackController<DataTypes>::onMouseEvent(core::objectmodel::MouseEvent * /*mev*/)
{
}

template <>
void ARTrackController<RigidTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
    switch (mev->getState())
    {
    case core::objectmodel::MouseEvent::RightPressed :
        rightPressed = true;
        break;
    case core::objectmodel::MouseEvent::RightReleased :
        rightPressed = false;
        break;
    case core::objectmodel::MouseEvent::LeftPressed :
        beginLocalPosition = Vec3d(-mev->getPosX(), 0, mev->getPosY());
        leftPressed = true;
        break;
    case core::objectmodel::MouseEvent::LeftReleased :
        leftPressed = false;
        break;
    case core::objectmodel::MouseEvent::Reset :
        leftPressed = false;
        rightPressed = false;
        break;
    default:
        break;
    }

    if(leftPressed)
    {
        endLocalPosition = Vec3d(-mev->getPosX(), 0, mev->getPosY());
        (*mstate->getXfree())[0].getCenter() += (endLocalPosition - beginLocalPosition);
        (*mstate->getX())[0].getCenter() += (endLocalPosition - beginLocalPosition);
        beginLocalPosition = endLocalPosition;
    }
    if(rightPressed)
    {
        //TODO: build a quat using mouse events to turn the hand.
        (*mstate->getXfree())[0].getOrientation() = Quat(mev->getPosX(), 0, 0, 1);
        (*mstate->getX())[0].getOrientation() = Quat(mev->getPosX(), 0, 0, 1);
    }
}

template <class DataTypes>
void ARTrackController<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::core::objectmodel::ARTrackEvent *>(event))
    {
        sofa::core::objectmodel::ARTrackEvent *aev = dynamic_cast<sofa::core::objectmodel::ARTrackEvent *>(event);
        onARTrackEvent(aev);
    }

    if (dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event))
    {
        sofa::core::objectmodel::MouseEvent *mev = dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event);
        onMouseEvent(mev);
    }
}


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_H
