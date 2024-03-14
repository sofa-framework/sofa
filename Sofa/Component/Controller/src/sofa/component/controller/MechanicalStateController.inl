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
#pragma once
#include <sofa/component/controller/MechanicalStateController.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Quat.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalProjectPositionAndVelocityVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor;

namespace sofa::component::controller
{

template <class DataTypes>
MechanicalStateController<DataTypes>::MechanicalStateController()
    : index( initData(&index, (unsigned int)0, "index", "Index of the controlled DOF") )
    , onlyTranslation( initData(&onlyTranslation, false, "onlyTranslation", "Controlling the DOF only in translation") )
    , buttonDeviceState(initData(&buttonDeviceState, false, "buttonDeviceState", "state of ths device button"))
    , mainDirection( initData(&mainDirection, sofa::type::Vec<3,Real>((Real)0.0, (Real)0.0, (Real)-1.0), "mainDirection", "Main direction and orientation of the controlled DOF") )
{
    mainDirection.beginEdit()->normalize();
    mainDirection.endEdit();
}

template <class DataTypes>
void MechanicalStateController<DataTypes>::init()
{
    using core::behavior::MechanicalState;
    mState = dynamic_cast<MechanicalState<DataTypes> *> (this->getContext()->getMechanicalState());
    
    msg_error_when(!mState) << "MechanicalStateController has no binding MechanicalState";
    device = false;
}


template <class DataTypes>
void MechanicalStateController<DataTypes>::applyController()
{
    using sofa::type::Quat;
    using sofa::type::Vec;

    if(device)
    {
        if(mState)
        {
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
            mState->vRealloc( sofa::core::mechanicalparams::defaultInstance(), core::VecCoordId::freePosition() ); // freePosition is not allocated by default
            helper::WriteAccessor<Data<VecCoord> > xfree = *this->mState->write(core::VecCoordId::freePosition());

            unsigned int i = index.getValue();

            Vec<3,Real> vx(1,0,0);
            Vec<3,Real> vy(0,1,0);
            Vec<3,Real> vz(0,0,1);

            if (mouseMode==BtLeft)
            {
                xfree[i].getOrientation() = x[i].getOrientation() * Quat<SReal>(vy, dx * (Real)0.001) * Quat<SReal>(vz, dy * (Real)0.001);
                x[i].getOrientation() = x[i].getOrientation() * Quat<SReal>(vy, dx * (Real)0.001) * Quat<SReal>(vz, dy * (Real)0.001);
            }
            else
            {
                sofa::type::Quat<Real>& quatrot = x[i].getOrientation();
                sofa::type::Vec<3,Real> vectrans(dy * mainDirection.getValue()[0] * (Real)0.05, dy * mainDirection.getValue()[1] * (Real)0.05, dy * mainDirection.getValue()[2] * (Real)0.05);
                vectrans = quatrot.rotate(vectrans);

                x[i].getCenter() += vectrans;
                x[i].getOrientation() = x[i].getOrientation() * Quat<SReal>(vx, dx * (Real)0.001);

                if(xfree.size() > 0)
                {
                    xfree[i].getCenter() += vectrans;
                    xfree[i].getOrientation() = x[i].getOrientation() * Quat<SReal>(vx, dx * (Real)0.001);
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

    auto node = this->getContext();
    MechanicalProjectPositionAndVelocityVisitor mechaProjectVisitor(core::mechanicalparams::defaultInstance()); mechaProjectVisitor.execute(node);
    MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(core::mechanicalparams::defaultInstance()); mechaVisitor.execute(node);
    sofa::simulation::UpdateMappingVisitor updateVisitor(core::execparams::defaultInstance()); updateVisitor.execute(node);
}


template <class DataTypes>
void MechanicalStateController<DataTypes>::onBeginAnimationStep(const double /*dt*/)
{
    buttonDevice=buttonDeviceState.getValue();
    applyController();
}



template <class DataTypes>
core::behavior::MechanicalState<DataTypes> *MechanicalStateController<DataTypes>::getMechanicalState() const
{
    return mState;
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::setMechanicalState(core::behavior::MechanicalState<DataTypes> *_mState)
{
    mState = _mState;
}



template <class DataTypes>
unsigned int MechanicalStateController<DataTypes>::getIndex() const
{
    return index.getValue();
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::setIndex(const unsigned int _index)
{
    index.setValue(_index);
}



template <class DataTypes>
const sofa::type::Vec<3, typename MechanicalStateController<DataTypes>::Real > &MechanicalStateController<DataTypes>::getMainDirection() const
{
    return mainDirection.getValue();
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::setMainDirection(const sofa::type::Vec<3,Real> _mainDirection)
{
    mainDirection.setValue(_mainDirection);
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::onMouseEvent(core::objectmodel::MouseEvent* /*mev*/)
{

}

} // namespace sofa::component::controller
