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

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <iostream>


namespace sofa::core::behavior
{

template<class DataTypes>
ProjectiveConstraintSet<DataTypes>::ProjectiveConstraintSet(MechanicalState<DataTypes> *mm)
    : Inherit1(), Inherit2(mm)
    , endTime( initData(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
{
}

template<class DataTypes>
ProjectiveConstraintSet<DataTypes>::~ProjectiveConstraintSet()
{
}

template <class DataTypes>
bool ProjectiveConstraintSet<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectJacobianMatrix(const MechanicalParams* mparams, MultiMatrixDerivId cId)
{
    if (!isActive())
        return;

    if (this->mstate)
    {
        projectJacobianMatrix(mparams, *cId[this->mstate.get()].write());
    }
}



template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId)
{
    if (!isActive())
        return;
    if (this->mstate)
    {
            projectResponse(mparams, *dxId[this->mstate.get()].write());
    }
    msg_error_when(!this->mstate) << "ProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId), no this->mstate for " << this->getName();
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId)
{
    if (!isActive())
        return;

    if (this->mstate)
    {

            projectVelocity(mparams, *vId[this->mstate.get()].write());
    }
    msg_error_when(!this->mstate) << "ProjectiveConstraintSet<DataTypes>::projectVelocity(const MechanicalParams* mparams, MultiVecDerivId dxId), no this->mstate for " << this->getName();
}

template<class DataTypes>
void ProjectiveConstraintSet<DataTypes>::projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId)
{
    if (!isActive())
        return;

    if (this->mstate)
    {

            projectPosition(mparams, *xId[this->mstate.get()].write());
    }
}


} // namespace sofa::core::behavior
