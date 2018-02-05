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
#ifndef SOFA_CORE_BEHAVIOR_PAIRINTERACTIONCONSTRAINT_INL
#define SOFA_CORE_BEHAVIOR_PAIRINTERACTIONCONSTRAINT_INL

#include <sofa/core/behavior/PairInteractionConstraint.h>


namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
PairInteractionConstraint<DataTypes>::PairInteractionConstraint(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : endTime( initData(&endTime,(SReal)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
    , mstate1(initLink("object1", "First object to constrain"), mm1)
    , mstate2(initLink("object2", "Second object to constrain"), mm2)
{
}

template<class DataTypes>
PairInteractionConstraint<DataTypes>::~PairInteractionConstraint()
{
}

template<class DataTypes>
void PairInteractionConstraint<DataTypes>::init()
{
    BaseInteractionConstraint::init();

    if (mstate1 == NULL || mstate2 == NULL)
    {
        mstate1 = mstate2 = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    }
}

template<class DataTypes>
bool PairInteractionConstraint<DataTypes>::isActive() const
{
    if (endTime.getValue() < 0)
        return true;

    return endTime.getValue() > getContext()->getTime();
}


template<class DataTypes>
void PairInteractionConstraint<DataTypes>::getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v)
{
    if (cParams)
    {
        getConstraintViolation(cParams, v, *cParams->readX(mstate1), *cParams->readX(mstate2), *cParams->readV(mstate1), *cParams->readV(mstate2));
    }
}


template<class DataTypes>
void PairInteractionConstraint<DataTypes>::buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex)
{
    if (cParams)
    {
        buildConstraintMatrix(cParams, *cId[mstate1.get(cParams)].write(), *cId[mstate2.get(cParams)].write(), cIndex, *cParams->readX(mstate1), *cParams->readX(mstate2));
        updateForceMask();
    }
}

template<class DataTypes>
void PairInteractionConstraint<DataTypes>::updateForceMask()
{
    // the default implementation adds every dofs to the mask
    // this sould be overloaded by each forcefield to only add the implicated dofs subset to the mask
    mstate1->forceMask.assign( mstate1->getSize(), true );
    mstate2->forceMask.assign( mstate2->getSize(), true );
}

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
