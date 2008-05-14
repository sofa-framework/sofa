/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Models: MechanicalStateController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_INL
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_INL

#include <sofa/component/controller/MechanicalStateController.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/simulation/tree/UpdateMappingVisitor.h>



namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
MechanicalStateController<DataTypes>::MechanicalStateController()
    : index( initData(&index, (unsigned int)0, "index", "Index of the controlled DOF") )
    , mainDirection(sofa::defaulttype::Vec<3,Real>((Real)0.0, (Real)0.0, (Real)-1.0))
    , mainDirectionPtr( initDataPtr(&mainDirectionPtr, &mainDirection, "mainDirection", "Main direction and orientation of the controlled DOF") )
{
    mainDirection.normalize();
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::init()
{
    using sofa::simulation::tree::GNode;
    using core::componentmodel::behavior::MechanicalState;

    mState = dynamic_cast<MechanicalState<DataTypes> *> (this->getContext()->getMechanicalState());
    if (!mState)
        std::cerr << "WARNING - MechanicalStateController has no binding MechanicalState\n";
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::onMouseEvent(core::objectmodel::MouseEvent *mev)
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

    case sofa::core::objectmodel::MouseEvent::Move :
        if (handleEventTriggersUpdate.getValue())
            applyController();
        break;

    default :
        break;
    }
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::onBeginAnimationStep()
{
    applyController();
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::applyController()
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;

    if ((mouseMode==BtLeft) || (mouseMode==BtRight))
    {
        int dx = eventX - mouseSavedPosX;
        int dy = eventY - mouseSavedPosY;
        mouseSavedPosX = eventX;
        mouseSavedPosY = eventY;

        if (mState)
        {
            unsigned int i = index.getValue();

            Vec<3,Real> x(1,0,0);
            Vec<3,Real> y(0,1,0);
            Vec<3,Real> z(0,0,1);

            if (mouseMode==BtLeft)
            {
                (*mState->getXfree())[i].getOrientation() = (*mState->getX())[i].getOrientation() * Quat(y, dx * (Real)0.001) * Quat(z, dy * (Real)0.001);
                (*mState->getX())[i].getOrientation() = (*mState->getX())[i].getOrientation() * Quat(y, dx * (Real)0.001) * Quat(z, dy * (Real)0.001);
            }
            else
            {
                sofa::helper::Quater<Real>& quatrot = (*mState->getX())[i].getOrientation();
                sofa::defaulttype::Vec<3,Real> vectrans(dy * mainDirection[0] * (Real)0.05, dy * mainDirection[1] * (Real)0.05, dy * mainDirection[2] * (Real)0.05);
                vectrans = quatrot.rotate(vectrans);
                (*mState->getXfree())[i].getCenter() += vectrans;
                (*mState->getX())[i].getCenter() += vectrans;

                (*mState->getXfree())[i].getOrientation() = (*mState->getX())[i].getOrientation() * Quat(x, dx * (Real)0.001);
                (*mState->getX())[i].getOrientation() = (*mState->getX())[i].getOrientation() * Quat(x, dx * (Real)0.001);
            }
        }
        sofa::simulation::tree::GNode *node = static_cast<sofa::simulation::tree::GNode*> (this->getContext());
        sofa::simulation::tree::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor; mechaVisitor.execute(node);
        sofa::simulation::tree::UpdateMappingVisitor updateVisitor; updateVisitor.execute(node);
    }
}



template <class DataTypes>
core::componentmodel::behavior::MechanicalState<DataTypes> *MechanicalStateController<DataTypes>::getMechanicalState() const
{
    return mState;
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::setMechanicalState(core::componentmodel::behavior::MechanicalState<DataTypes> *_mState)
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
const sofa::defaulttype::Vec<3, typename MechanicalStateController<DataTypes>::Real > &MechanicalStateController<DataTypes>::getMainDirection() const
{
    return mainDirection;
}



template <class DataTypes>
void MechanicalStateController<DataTypes>::setMainDirection(const sofa::defaulttype::Vec<3,Real> _mainDirection)
{
    mainDirection = _mainDirection;
}


} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_H
