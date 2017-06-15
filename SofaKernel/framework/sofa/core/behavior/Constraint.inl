/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_CORE_BEHAVIOR_CONSTRAINT_INL
#define SOFA_CORE_BEHAVIOR_CONSTRAINT_INL

#include <sofa/core/behavior/Constraint.h>

namespace sofa
{

namespace core
{

namespace behavior
{

template<class DataTypes>
Constraint<DataTypes>::Constraint(MechanicalState<DataTypes> *mm)
    : endTime( initData(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
    , mstate(mm)
{
}

template<class DataTypes>
Constraint<DataTypes>::~Constraint()
{
}


template <class DataTypes>
bool Constraint<DataTypes>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}


template<class DataTypes>
void Constraint<DataTypes>::init()
{
    BaseConstraint::init();
    mstate = dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
}


template<class DataTypes>
void Constraint<DataTypes>::getConstraintViolation(const ConstraintParams* cParams, defaulttype::BaseVector *v)
{
    if (cParams)
    {
        getConstraintViolation(cParams, v, *cParams->readX(mstate), *cParams->readV(mstate));
    }
}


template<class DataTypes>
void Constraint<DataTypes>::buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex)
{
    if (cParams)
    {
        buildConstraintMatrix(cParams, *cId[mstate].write(), cIndex, *cParams->readX(mstate));
        updateForceMask();
    }
}

template<class DataTypes>
void Constraint<DataTypes>::updateForceMask()
{
    // the default implementation adds every dofs to the mask
    // this sould be overloaded by each forcefield to only add the implicated dofs subset to the mask
    mstate->forceMask.assign( mstate->getSize(), true );
}


} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_CONSTRAINT_INL
