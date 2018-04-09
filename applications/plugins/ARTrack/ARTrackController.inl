/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_INL
#define SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_INL

#include "ARTrackController.h"
#include "ARTrackEvent.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/VecId.h>

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
void ARTrackController<DataTypes>::init()
{

    std::cout<<" ARTrackController<DataTypes>::init"<<std::endl;
}


#ifndef SOFA_FLOAT
template <>
void ARTrackController<Vec1dTypes>::init()
{
    getContext()->get<sofa::component::container::Articulation>(&articulations);
}

template <>
void ARTrackController<Vec3dTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    if(mstate)
    {
        if(!mstate->read(core::ConstVecCoordId::freePosition())->getValue().empty() && !mstate->read(core::ConstVecCoordId::position())->getValue().empty())
        {
            helper::WriteAccessor<Data<VecCoord> > freePos = *mstate->write(core::VecCoordId::freePosition());
            helper::WriteAccessor<Data<VecCoord> > pos = *mstate->write(core::VecCoordId::position());
            for (unsigned int i=0; i<3; ++i)
            {
                std::cout<<"finger["<<i<<"] = "<<aev->getFingerposition(i)<<std::endl;
                freePos[i] = aev->getFingerposition(i);
                pos[i] = aev->getFingerposition(i);
            }
        }
    }
}
#endif

template <>
void ARTrackController<RigidTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    std::cout<<"AR track event detected"<<std::endl;
    if(mstate)
    {
        if(!mstate->read(core::ConstVecCoordId::freePosition())->getValue().empty() && !mstate->read(core::ConstVecCoordId::position())->getValue().empty())
        {
            if(f_printLog.getValue())
                std::cout<<" aev pos :"<<aev->getPosition()<<" aev quat :"<<aev->getOrientation()<<std::endl;

            helper::WriteAccessor<Data<VecCoord> > freePos = *mstate->write(core::VecCoordId::freePosition());
            helper::WriteAccessor<Data<VecCoord> > pos = *mstate->write(core::VecCoordId::position());

            freePos[0].getCenter() = aev->getPosition();
            pos[0].getCenter() = aev->getPosition();

            freePos[0].getOrientation() = aev->getOrientation();
            pos[0].getOrientation() = aev->getOrientation();

            freePos[0].getOrientation().normalize();
            pos[0].getOrientation().normalize();

        }
    }
}

#ifndef SOFA_FLOAT
template <>
void ARTrackController<Vec1dTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{
    std::cout<<"AR track event detected"<<std::endl;
    if(mstate)
    {
        if(!mstate->read(core::ConstVecCoordId::freePosition())->getValue().empty() && !mstate->read(core::ConstVecCoordId::position())->getValue().empty())
        {

            if(f_printLog.getValue())
                std::cout<<"pouce :"<<aev->getAngles()[0]<<"  index:"<<aev->getAngles()[1]<<"  autres:"<<aev->getAngles()[2]<<std::endl;


            helper::WriteAccessor<Data<VecCoord> > restPos = *mstate->write(core::VecCoordId::restPosition());

            for (unsigned int i=6; i<9; ++i) // thumb
            {
                restPos[i].x() = (aev->getAngles()[0]);// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                /* if(mstate->read(core::ConstVecCoordId::restPosition())->getValue()[i].x()<0)
                 {
                     mstate->read(core::ConstVecCoordId::restPosition())->getValue()[i] = 0.0;
                     mstate->read(core::ConstVecCoordId::position())->getValue()[i] = 0.0;
                 }*/
            }

            for (unsigned int i=9; i<12; ++i) // index
            {
                restPos[i].x() = (aev->getAngles()[1]);// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                /* if(mstate->read(core::ConstVecCoordId::restPosition())->getValue()[i].x()<0)
                 {
                     mstate->read(core::ConstVecCoordId::restPosition())->getValue()[i] = 0.0;
                     mstate->read(core::ConstVecCoordId::position())->getValue()[i] = 0.0;
                 }*/
            }

            for(unsigned int i=12; i<21; ++i) // middle, ring, little.
            {
                restPos[i].x() = (aev->getAngles()[2]);// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();

                /*if(mstate->read(core::ConstVecCoordId::restPosition())->getValue()[i].x()<0)
                {
                    mstate->read(core::ConstVecCoordId::restPosition())->getValue()[i] = 0.0;
                    mstate->read(core::ConstVecCoordId::position())->getValue()[i] = 0.0;
                }*/
            }
        }
    }
}
#endif

template <class DataTypes>
void ARTrackController<DataTypes>::onARTrackEvent(core::objectmodel::ARTrackEvent* /*aev*/)
{
}

template <class DataTypes>
void ARTrackController<DataTypes>::onMouseEvent(core::objectmodel::MouseEvent * /*mev*/)
{
}

#ifndef SOFA_FLOAT
template <>
void ARTrackController<Vec1dTypes>::onMouseEvent(core::objectmodel::MouseEvent * mev)
{
    std::cout<<" onMouseEvent on Vec1Types called "<<std::endl;
    switch (mev->getState())
    {
    case core::objectmodel::MouseEvent::Wheel:
        wheel=true;
        break;
    default:
        break;
    }
    if(wheel)
    {
        int delta = mev->getWheelDelta() ;
        double Delta = ((double)delta )/3000.0;
        std::cout<<"Delta Wheel ="<<Delta<<std::endl;

        if(mstate)
        {
            if(!mstate->read(core::ConstVecCoordId::freePosition())->getValue().empty() && !mstate->read(core::ConstVecCoordId::position())->getValue().empty())
            {

                helper::WriteAccessor<Data<VecCoord> > restPos = *mstate->write(core::VecCoordId::restPosition());

                for (unsigned int i=6; i<9; ++i) // thumb
                {
                    restPos[i].x() += Delta;// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();


                }

                for (unsigned int i=9; i<12; ++i) // index
                {
                    restPos[i].x() += Delta;// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();


                }

                for(unsigned int i=12; i<21; ++i) // middle, ring, little.
                {
                    restPos[i].x() += Delta;// * articulations[i]->coeff.getValue() - articulations[i]->correction.getValue();


                }
            }
        }

        wheel=false;
    }
}
#endif

template <>
void ARTrackController<RigidTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{
    std::cout<<" onMouseEvent on RigidTypes called "<<std::endl;

    switch (mev->getState())
    {
    case core::objectmodel::MouseEvent::RightPressed :
        beginLocalPosition = Vec3d(0, -mev->getPosY(),0);
        rightPressed = true;
        break;
    case core::objectmodel::MouseEvent::RightReleased :
        rightPressed = false;
        break;
    case core::objectmodel::MouseEvent::LeftPressed :
        beginLocalPosition = Vec3d(mev->getPosX(), 0, mev->getPosY());
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

    helper::WriteAccessor<Data<VecCoord> > freePos = *mstate->write(core::VecCoordId::freePosition());
    helper::WriteAccessor<Data<VecCoord> > pos = *mstate->write(core::VecCoordId::position());

    if(leftPressed)
    {
        endLocalPosition = Vec3d(mev->getPosX(), 0, mev->getPosY());
        freePos[0].getCenter() += (endLocalPosition - beginLocalPosition);
        pos[0].getCenter() += (endLocalPosition - beginLocalPosition);
        beginLocalPosition = endLocalPosition;
    }
    if(rightPressed)
    {
        endLocalPosition = Vec3d(0, -mev->getPosY(),0);
        //TODO: build a quat using mouse events to turn the hand.
        freePos[0].getCenter() += (endLocalPosition - beginLocalPosition);
        pos[0].getCenter() +=(endLocalPosition- beginLocalPosition);
        beginLocalPosition = endLocalPosition;
    }

    freePos[0].getOrientation() = Quat( 0.0,0.0, sin(3.14/2), cos(3.14/2));
    pos[0].getOrientation() = Quat( 0.0,0.0, sin(3.14/2), cos(3.14/2));
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
