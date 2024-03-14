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

#include <sofa/core/behavior/MixedInteractionConstraint.h>
#include <sofa/core/ConstraintParams.h>

namespace sofa::core::behavior
{

template<class DataTypes1, class DataTypes2>
MixedInteractionConstraint<DataTypes1, DataTypes2>::MixedInteractionConstraint(MechanicalState<DataTypes1> *mm1, MechanicalState<DataTypes2> *mm2)
    : Inherit1(), Inherit2(mm1, mm2)
    , endTime( initData(&endTime,(SReal)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
{
}

template<class DataTypes1, class DataTypes2>
MixedInteractionConstraint<DataTypes1, DataTypes2>::~MixedInteractionConstraint()
{
}

template<class DataTypes1, class DataTypes2>
bool MixedInteractionConstraint<DataTypes1, DataTypes2>::isActive() const
{
    if( endTime.getValue()<0 ) return true;
    return endTime.getValue()>getContext()->getTime();
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionConstraint<DataTypes1, DataTypes2>::getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v)
{
    if (cParams)
    {
        getConstraintViolation(cParams, v, *cParams->readX(this->mstate1.get()), *cParams->readX(this->mstate2.get()), 
                                           *cParams->readV(this->mstate1.get()), *cParams->readV(this->mstate2.get()));
    }
}

template<class DataTypes1, class DataTypes2>
void MixedInteractionConstraint<DataTypes1, DataTypes2>::buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex)
{
    if (cParams)
    {
        buildConstraintMatrix(cParams, *cId[this->mstate1.get()].write(), *cId[this->mstate2.get()].write(), cIndex, 
                                        *cParams->readX(this->mstate1.get()), *cParams->readX(this->mstate2.get()));
    }
}

} // namespace sofa::core::behavior
