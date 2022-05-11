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

#include <sofa/core/behavior/PairInteractionProjectiveConstraintSet.h>


namespace sofa::core::behavior
{

template<class DataTypes>
PairInteractionProjectiveConstraintSet<DataTypes>::PairInteractionProjectiveConstraintSet(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : Inherit1(), Inherit2(mm1, mm2)
    , endTime( initData(&endTime,(SReal)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
{
    if (!mm1)
        this->mstate1.setPath("@./"); // default to state of the current node
    if (!mm2)
        this->mstate2.setPath("@./"); // default to state of the current node
}

template<class DataTypes>
PairInteractionProjectiveConstraintSet<DataTypes>::~PairInteractionProjectiveConstraintSet()
{
}

template<class DataTypes>
bool PairInteractionProjectiveConstraintSet<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectJacobianMatrix(const MechanicalParams* /*mparams*/, MultiMatrixDerivId /*cId*/)
{
    msg_error()<< "NOT IMPLEMENTED YET";
}



template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId)
{
    if( !isActive() ) return;
    if (this->mstate1 && this->mstate2)
    {
        projectResponse(mparams, *dxId[this->mstate1.get()].write(), *dxId[this->mstate2.get()].write());
    }
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId)
{
    if( !isActive() ) return;
    if (this->mstate1 && this->mstate2)
    {
        projectVelocity(mparams, *vId[this->mstate1.get()].write(), *vId[this->mstate2.get()].write());
    }
}

template<class DataTypes>
void PairInteractionProjectiveConstraintSet<DataTypes>::projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId)
{
    if( !isActive() ) return;
    if (this->mstate1 && this->mstate2)
    {
        projectPosition(mparams, *xId[this->mstate1.get()].write(), *xId[this->mstate2.get()].write());
    }
}

} // namespace sofa::core::behavior
