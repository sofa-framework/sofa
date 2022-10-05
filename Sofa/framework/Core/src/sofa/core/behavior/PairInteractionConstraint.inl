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

#include <sofa/core/behavior/PairInteractionConstraint.h>
#include <sofa/core/ConstraintParams.h>

namespace sofa::core::behavior
{

template<class DataTypes>
PairInteractionConstraint<DataTypes>::PairInteractionConstraint(MechanicalState<DataTypes> *mm1, MechanicalState<DataTypes> *mm2)
    : Inherit1(), Inherit2(mm1, mm2)
    , endTime( initData(&endTime,(SReal)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
{
}

template<class DataTypes>
PairInteractionConstraint<DataTypes>::~PairInteractionConstraint()
{
}

template<class DataTypes>
bool PairInteractionConstraint<DataTypes>::isActive() const
{
    if (endTime.getValue() < 0)
        return true;

    return endTime.getValue() > getContext()->getTime();
}


template<class DataTypes>
void PairInteractionConstraint<DataTypes>::getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v)
{
    if (cParams)
    {
        getConstraintViolation(cParams, v, *cParams->readX(this->mstate1.get()), *cParams->readX(this->mstate2.get()), *cParams->readV(this->mstate1.get()), *cParams->readV(this->mstate2.get()));
    }
}


template<class DataTypes>
void PairInteractionConstraint<DataTypes>::buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex)
{
    if (cParams)
    {
        buildConstraintMatrix(cParams, *cId[this->mstate1.get()].write(), *cId[this->mstate2.get()].write(), cIndex, *cParams->readX(this->mstate1.get()), *cParams->readX(this->mstate2.get()));
    }
}

template<class DataTypes>
void PairInteractionConstraint<DataTypes>::storeLambda(const ConstraintParams* cParams, MultiVecDerivId res, const sofa::linearalgebra::BaseVector* lambda)
{
    if (cParams)
    {
        storeLambda(cParams, *res[this->mstate1.get()].write(), *res[this->mstate2.get()].write(), *cParams->readJ(this->mstate1.get()), *cParams->readJ(this->mstate2.get()), lambda);
    }
}


template<class DataTypes>
void PairInteractionConstraint<DataTypes>::storeLambda(const ConstraintParams*, Data<VecDeriv>& result1, Data<VecDeriv>& result2,
    const Data<MatrixDeriv>& jacobian1, const Data<MatrixDeriv>& jacobian2, const sofa::linearalgebra::BaseVector* lambda)
{
    auto res1 = sofa::helper::getWriteAccessor(result1);
    auto res2 = sofa::helper::getWriteAccessor(result2);
    const MatrixDeriv& j1 = jacobian1.getValue();
    const MatrixDeriv& j2 = jacobian2.getValue();

    j1.multTransposeBaseVector(res1, lambda );
    j2.multTransposeBaseVector(res2, lambda );
}

} // namespace sofa::core::behavior
