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
//
// C++ Models: MechanicalStateControllerOmni
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLEROMNI_INL
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLEROMNI_INL

#include <SofaUserInteraction/MechanicalStateControllerOmni.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/MouseEvent.h>
//#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
MechanicalStateControllerOmni<DataTypes>::MechanicalStateControllerOmni()
    :/* index( initData(&index, (unsigned int)0, "index", "Index of the controlled DOF") )
    , onlyTranslation( initData(&onlyTranslation, false, "onlyTranslation", "Controlling the DOF only in translation") )
    ,*/ buttonDeviceState(initData(&buttonDeviceState, false, "buttonDeviceState", "state of ths device button"))
    , deviceId(initData(&deviceId, -1, "deviceId", "id of active device for this controller"))
    , angle(initData(&angle, (Real)10, "angle", "max angle"))
    , speed(initData(&speed, (Real)30, "speed", "closing/opening speed"))
    //   , mainDirection( initData(&mainDirection, sofa::defaulttype::Vec<3,Real>((Real)0.0, (Real)0.0, (Real)-1.0), "mainDirection", "Main direction and orientation of the controlled DOF") )
{
    //mainDirection.beginEdit()->normalize();
    //mainDirection.endEdit();
}

template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::init()
{
    using core::behavior::MechanicalState;
    mState = dynamic_cast<MechanicalState<DataTypes> *> (this->getContext()->getMechanicalState());
    if (!mState)
        serr << "MechanicalStateControllerOmni has no binding MechanicalState" << sendl;
    device = false;
}


template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::applyController(double /*dt*/)
{
    /*
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;

    if(device)
    {
           if(mState)device
        {
    //			if(mState->read(sofa::core::ConstVecCoordId::freePosition())->getValue())
            {
                   helper::WriteAccessor<Data<VecCoord> > x = *this->mState->write(core::VecCoordId::position());
                   helper::WriteAccessor<Data<VecCoord> > xfree = *this->mState->write(core::VecCoordId::freePosition());
                xfree[0].getCenter() = position;
                x[0].getCenter() = position;

                xfree[0].getOrientation() = orientation;
                x[0].getOrientation() = orientation;
            }
        }
        device = false;
    }

    if ( !onlyTranslation.getValue()  && ((mouseMode==BtLeft) || (mouseMode==BtRight)))
    {
        int dx = eventX - mouseSavedPosX;
        int dy = eventY - mouseSavedPosY;
        mouseSavedPosX = eventX;
        mouseSavedPosY = eventY;

        if (mState)
        {
               helper::WriteAccessor<Data<VecCoord> > x = *this->mState->write(core::VecCoordId::position());
               helper::WriteAccessor<Data<VecCoord> > xfree = *this->mState->write(core::VecCoordId::freePosition());

            unsigned int i = index.getValue();

            Vec<3,Real> vx(1,0,0);
            Vec<3,Real> vy(0,1,0);
            Vec<3,Real> vz(0,0,1);

            if (mouseMode==BtLeft)
            {
                xfree[i].getOrientation() = x[i].getOrientation() * Quat(vy, dx * (Real)0.001) * Quat(vz, dy * (Real)0.001);
                x[i].getOrientation() = x[i].getOrientation() * Quat(vy, dx * (Real)0.001) * Quat(vz, dy * (Real)0.001);
            }
            else
            {
                sofa::helper::Quater<Real>& quatrot = x[i].getOrientation();
                sofa::defaulttype::Vec<3,Real> vectrans(dy * mainDirection.getValue()[0] * (Real)0.05, dy * mainDirection.getValue()[1] * (Real)0.05, dy * mainDirection.getValue()[2] * (Real)0.05);
                vectrans = quatrot.rotate(vectrans);

                x[i].getCenter() += vectrans;
                x[i].getOrientation() = x[i].getOrientation() * Quat(vx, dx * (Real)0.001);

            //	x0[i].getCenter() += vectrans;
            //	x0[i].getOrientation() = x0[i].getOrientation() * Quat(vx, dx * (Real)0.001);

                if(xfree.size() > 0)
                {
                    xfree[i].getCenter() += vectrans;
                    xfree[i].getOrientation() = x[i].getOrientation() * Quat(vx, dx * (Real)0.001);
                }
            }
        }
    }
    else if( onlyTranslation.getValue() )
    {
        if( mouseMode )
        {
            int dx = eventX - mouseSavedPosX;
            int dy = eventY - mouseSavedPosY;
            mouseSavedPosX = eventX;
            mouseSavedPosY = eventY;

    // 			Real d = sqrt(dx*dx+dy*dy);
    // 			if( dx<0 || dy<0 ) d = -d;

            if (mState)
            {
                   helper::WriteAccessor<Data<VecCoord> > x = *this->mState->write(core::VecCoordId::position());

                unsigned int i = index.getValue();

                switch( mouseMode )
                {
                    case BtLeft:
                        x[i].getCenter() += Vec<3,Real>((Real)dx,(Real)0,(Real)0);
                        break;
                    case BtRight :
                        x[i].getCenter() += Vec<3,Real>((Real)0,(Real)dy,(Real)0);
                        break;
                    case BtMiddle :
                        x[i].getCenter() += Vec<3,Real>((Real)0,(Real)0,(Real)dy);
                        break;
                    default :
                        break;
                }
            }
        }
    }

    sofa::simulation::Node *node = static_cast<sofa::simulation::Node*> (this->getContext());
       sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
       sofa::simulation::UpdateMappingVisitor updateVisitor(core::ExecParams::defaultInstance()); updateVisitor.execute(node);
       */
};



template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::onHapticDeviceEvent(core::objectmodel::HapticDeviceEvent *oev)
{
    if (deviceId.getValue() != -1 && (int)oev->getDeviceId() != deviceId.getValue()) return;

    device = true;
    buttonDeviceState.setValue(oev->getButton());
    position = oev->getPosition();
    orientation = oev->getOrientation();
}

template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::onBeginAnimationStep(const double dt)
{
    applyController(dt);
}



template <class DataTypes>
core::behavior::MechanicalState<DataTypes> *MechanicalStateControllerOmni<DataTypes>::getMechanicalState() const
{
    return mState;
}



template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::setMechanicalState(core::behavior::MechanicalState<DataTypes> *_mState)
{
    mState = _mState;
}


/*
template <class DataTypes>
unsigned int MechanicalStateControllerOmni<DataTypes>::getIndex() const
{
    return index.getValue();
}



template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::setIndex(const unsigned int _index)
{
    index.setValue(_index);
}



template <class DataTypes>
const sofa::defaulttype::Vec<3, typename MechanicalStateControllerOmni<DataTypes>::Real > &MechanicalStateControllerOmni<DataTypes>::getMainDirection() const
{
    return mainDirection.getValue();
}



template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::setMainDirection(const sofa::defaulttype::Vec<3,Real> _mainDirection)
{
    mainDirection.setValue(_mainDirection);
}
*/

/*
template <class DataTypes>
void MechanicalStateControllerOmni<DataTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
{

}
*/


#ifndef SOFA_FLOAT
template <>
SOFA_USER_INTERACTION_API void MechanicalStateControllerOmni<defaulttype::Vec1dTypes>::applyController(double dt);
/*
template <>
SOFA_USER_INTERACTION_API void MechanicalStateControllerOmni<defaulttype::Vec1dTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev);

template <>
SOFA_USER_INTERACTION_API void MechanicalStateControllerOmni<defaulttype::Rigid3dTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev);
*/
#endif

#ifndef SOFA_DOUBLE
template <>
SOFA_USER_INTERACTION_API void MechanicalStateControllerOmni<defaulttype::Vec1fTypes>::applyController(double dt);
/*
template <>
SOFA_USER_INTERACTION_API void MechanicalStateControllerOmni<defaulttype::Vec1fTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev);

template <>
SOFA_USER_INTERACTION_API void MechanicalStateControllerOmni<defaulttype::Rigid3fTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev);
*/
#endif


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLEROMNI_INL
