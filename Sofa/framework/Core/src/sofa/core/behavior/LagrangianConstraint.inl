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

#include <sofa/core/behavior/Constraint.h>
#include <sofa/core/ConstraintParams.h>

namespace sofa::core::behavior
{

template<class DataTypes>
Constraint<DataTypes>::Constraint(MechanicalState<DataTypes> *mm)
    : Inherit1(), Inherit2(mm)
    , endTime( initData(&endTime,(Real)-1,"endTime","The constraint stops acting after the given value.\nUse a negative value for infinite constraints") )
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

template <class DataTypes>
void Constraint<DataTypes>::init()
{
    Inherit1::init();
    Inherit2::init();
}

template<class DataTypes>
void Constraint<DataTypes>::getConstraintViolation(const ConstraintParams* cParams, linearalgebra::BaseVector *v)
{
    if (cParams)
    {
        getConstraintViolation(cParams, v, *cParams->readX(this->mstate.get()), *cParams->readV(this->mstate.get()));
    }
}


template<class DataTypes>
void Constraint<DataTypes>::buildConstraintMatrix(const ConstraintParams* cParams, MultiMatrixDerivId cId, unsigned int &cIndex)
{
    if (cParams)
    {
        buildConstraintMatrix(cParams, *cId[this->mstate.get()].write(), cIndex, *cParams->readX(this->mstate.get()));
    }
}


template<class DataTypes>
void Constraint<DataTypes>::storeLambda(const ConstraintParams* cParams, MultiVecDerivId res, const sofa::linearalgebra::BaseVector* lambda)
{
    if (cParams)
    {
        storeLambda(cParams, *res[this->mstate.get()].write(), *cParams->readJ(this->mstate.get()), lambda);
    }
}

template<class DataTypes>
void Constraint<DataTypes>::storeLambda(const ConstraintParams*, Data<VecDeriv>& result, const Data<MatrixDeriv>& jacobian, const sofa::linearalgebra::BaseVector* lambda)
{
    auto res = sofa::helper::getWriteAccessor(result);
    const MatrixDeriv& j = jacobian.getValue();
    j.multTransposeBaseVector(res, lambda ); // lambda is a vector of scalar value so block size is one.
}


} // namespace sofa::core::behavior
