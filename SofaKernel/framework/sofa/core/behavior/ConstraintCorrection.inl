/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_CONSTRAINTCORRECTION_INL
#define SOFA_CORE_BEHAVIOR_CONSTRAINTCORRECTION_INL

#include <sofa/core/behavior/ConstraintCorrection.h>
#include <sofa/core/behavior/ConstraintSolver.h>


namespace sofa
{

namespace core
{

namespace behavior
{


template< class DataTypes >
void ConstraintCorrection< DataTypes >::init()
{
    mstate = dynamic_cast< behavior::MechanicalState< DataTypes >* >(getContext()->getMechanicalState());
}

template< class DataTypes >
void ConstraintCorrection< DataTypes >::cleanup()
{
    while(!constraintsolvers.empty())
    {
        constraintsolvers.back()->removeConstraintCorrection(this);
        constraintsolvers.pop_back();
    }
    sofa::core::behavior::BaseConstraintCorrection::cleanup();
}

template <class DataTypes>
void ConstraintCorrection<DataTypes>::addConstraintSolver(core::behavior::ConstraintSolver *s)
{
    constraintsolvers.push_back(s);
}

template <class DataTypes>
void ConstraintCorrection<DataTypes>::removeConstraintSolver(core::behavior::ConstraintSolver *s)
{
    constraintsolvers.remove(s);
}

template< class DataTypes >
void ConstraintCorrection< DataTypes >::computeAndApplyMotionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId v, core::MultiVecDerivId f, const defaulttype::BaseVector * lambda)
{
    if (mstate)
    {
        Data< VecCoord > *x_d = x[mstate].write();
        Data< VecDeriv > *v_d = v[mstate].write();
        Data< VecDeriv > *f_d = f[mstate].write();

        if (x_d && v_d && f_d)
        {
            computeAndApplyMotionCorrection(cparams, *x_d, *v_d, *f_d, lambda);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::computeAndApplyPositionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId res, core::MultiVecDerivId f, const defaulttype::BaseVector * lambda)
{
    if (mstate)
    {
        Data< VecCoord > *res_d = res[mstate].write();
        Data< VecDeriv > *f_d = f[mstate].write();

        if (res_d && f_d)
        {
            computeAndApplyPositionCorrection(cparams, *res_d, *f_d, lambda);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::computeAndApplyVelocityCorrection(const core::ConstraintParams *cparams, core::MultiVecDerivId res, core::MultiVecDerivId f, const defaulttype::BaseVector *lambda)
{
    if (mstate)
    {
        Data< VecDeriv > *res_d = res[mstate].write();
        Data< VecDeriv > *f_d = f[mstate].write();

        if (res_d && f_d)
        {
            computeAndApplyVelocityCorrection(cparams, *res_d, *f_d, lambda);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::applyPredictiveConstraintForce(const core::ConstraintParams *cparams, core::MultiVecDerivId f, const defaulttype::BaseVector *lambda)
{
    if (mstate)
    {
        Data< VecDeriv > *f_d = f[mstate].write();

        if (f_d)
        {
            applyPredictiveConstraintForce(cparams, *f_d, lambda);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::setConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector *lambda)
{
    if (mstate)
    {
        Data< VecDeriv > *f_d = f[mstate].write();

        if (f_d)
        {
            setConstraintForceInMotionSpace(*f_d, lambda);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::setConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector *lambda)
{
    VecDeriv& force = *f.beginEdit();

    const size_t numDOFs = mstate->getSize();

    force.clear();
    force.resize(numDOFs);
    for (size_t i = 0; i < numDOFs; i++)
        force[i] = Deriv();

    f.endEdit();

    addConstraintForceInMotionSpace(f, lambda);
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::addConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector *lambda)
{
    if (mstate)
    {
        Data< VecDeriv > *f_d = f[mstate].write();

        if (f_d)
        {
            addConstraintForceInMotionSpace(*f_d, lambda);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::addConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector *lambda)
{
    VecDeriv& force = *f.beginEdit();

    const size_t numDOFs = mstate->getSize();
    const size_t fPrevSize = force.size();

    if (numDOFs > fPrevSize)
    {
        force.resize(numDOFs);
        for (size_t i = fPrevSize; i < numDOFs; i++)
            force[i] = Deriv();
    }

    const MatrixDeriv& c = mstate->read(ConstMatrixDerivId::constraintJacobian())->getValue();

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const double lambdaC1 = lambda->element(rowIt.index());

        if (lambdaC1 != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                force[colIt.index()] += colIt.val() * lambdaC1;
            }
        }
    }

    f.endEdit();
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::setConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector *lambda, std::list< int > &activeDofs)
{
    if (mstate)
    {
        Data< VecDeriv > *f_d = f[mstate].write();

        if (f_d)
        {
            setConstraintForceInMotionSpace(*f_d, lambda, activeDofs);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::setConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector *lambda, std::list< int > &activeDofs)
{
    VecDeriv& force = *f.beginEdit();

    const size_t numDOFs = mstate->getSize();

    force.clear();
    force.resize(numDOFs);
    for (size_t i = 0; i < numDOFs; i++)
        force[i] = Deriv();

    f.endEdit();

    addConstraintForceInMotionSpace(f, lambda, activeDofs);
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::addConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector *lambda, std::list< int > &activeDofs)
{
    if (mstate)
    {
        Data< VecDeriv > *f_d = f[mstate].write();

        if (f_d)
        {
            addConstraintForceInMotionSpace(*f_d, lambda, activeDofs);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::addConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector *lambda, std::list< int > &activeDofs)
{
    VecDeriv& force = *f.beginEdit();

    const size_t numDOFs = mstate->getSize();
    const size_t fPrevSize = force.size();

    if (numDOFs > fPrevSize)
    {
        force.resize(numDOFs);
        for (size_t i = fPrevSize; i < numDOFs; i++)
            force[i] = Deriv();
    }

    const MatrixDeriv& c = mstate->read(ConstMatrixDerivId::constraintJacobian())->getValue();

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const double lambdaC1 = lambda->element(rowIt.index());

        if (lambdaC1 != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                force[colIt.index()] += colIt.val() * lambdaC1;
                activeDofs.push_back(colIt.index());
            }
        }
    }

    f.endEdit();

    activeDofs.sort();
    activeDofs.unique();
}


} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_CONSTRAINTCORRECTION_INL
