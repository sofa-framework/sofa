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

#include <sofa/core/behavior/ConstraintCorrection.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/ConstraintParams.h>

namespace sofa::core::behavior
{

template< class DataTypes >
void ConstraintCorrection< DataTypes >::init()
{
    Inherit2::init();

    if (!mstate)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

template< class DataTypes >
void ConstraintCorrection< DataTypes >::cleanup()
{
    for (auto it = l_constraintsolvers.rbegin(); it != l_constraintsolvers.rend(); ++it)
    {
        (*it)->removeConstraintCorrection(this);
    }
    l_constraintsolvers.clear();
    sofa::core::behavior::BaseConstraintCorrection::cleanup();
}

template <class DataTypes>
void ConstraintCorrection<DataTypes>::addConstraintSolver(core::behavior::ConstraintSolver *s)
{
    if (!l_constraintsolvers.add(s))
    {
        dmsg_error_when(!s) << "Trying to add an invalid constraint solver";
        dmsg_error() << "Cannot add the requested constraint solver";
    }
}

template <class DataTypes>
void ConstraintCorrection<DataTypes>::removeConstraintSolver(core::behavior::ConstraintSolver *s)
{
    if (!l_constraintsolvers.remove(s))
    {
        dmsg_error_when(!s) << "Trying to remove an invalid constraint solver";
        dmsg_error() << "Cannot remove the requested constraint solver";
    }
}

template< class DataTypes >
void ConstraintCorrection< DataTypes >::computeMotionCorrectionFromLambda(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, const linearalgebra::BaseVector * lambda)
{
    addConstraintForceInMotionSpace(cparams, cparams->lambda(), cparams->j(), lambda);

    computeMotionCorrection(cparams, dx, cparams->lambda());
}

template< class DataTypes >
void ConstraintCorrection< DataTypes >::applyMotionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId v, core::MultiVecDerivId dx, core::ConstMultiVecDerivId correction)
{
    if (mstate)
    {
        Data< VecCoord > *x_d  = x[mstate.get()].write();
        Data< VecDeriv > *v_d  = v[mstate.get()].write();
        Data< VecDeriv > *dx_d = dx[mstate.get()].write();
        const Data< VecDeriv > *correction_d = correction[mstate.get()].read();

        if (x_d && v_d && dx_d && correction_d)
        {
            applyMotionCorrection(cparams, *x_d, *v_d, *dx_d, *correction_d);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::applyPositionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId dx, core::ConstMultiVecDerivId correction)
{
    if (mstate)
    {
        Data< VecCoord > *x_d  = x[mstate.get()].write();
        Data< VecDeriv > *dx_d = dx[mstate.get()].write();
        const Data< VecDeriv > *correction_d = correction[mstate.get()].read();

        if (x_d && dx_d && correction_d)
        {
            applyPositionCorrection(cparams, *x_d, *dx_d, *correction_d);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::applyVelocityCorrection(const core::ConstraintParams *cparams, core::MultiVecDerivId v, core::MultiVecDerivId dv, core::ConstMultiVecDerivId correction)
{
    if (mstate)
    {
        Data< VecDeriv >* v_d  = v[mstate.get()].write();
        Data< VecDeriv >* dv_d = dv[mstate.get()].write();
        const Data< VecDeriv >* correction_d = correction[mstate.get()].read();

        if (v_d && dv_d && correction_d)
        {
            applyVelocityCorrection(cparams, *v_d, *dv_d, *correction_d);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::applyPredictiveConstraintForce(const core::ConstraintParams *cparams, core::MultiVecDerivId f, const linearalgebra::BaseVector *lambda)
{
    if (mstate)
    {
        addConstraintForceInMotionSpace(cparams, f, cparams->j(), lambda);
    }
}

template< class DataTypes >
void ConstraintCorrection< DataTypes >::addConstraintForceInMotionSpace(const core::ConstraintParams* cparams, core::MultiVecDerivId f, core::ConstMultiMatrixDerivId j, const linearalgebra::BaseVector * lambda)
{
    if (mstate)
    {
        Data< VecDeriv > *f_d = f[mstate.get()].write();
        const Data< MatrixDeriv > * j_d = j[mstate.get()].read();
        if (f_d && j_d)
        {
            addConstraintForceInMotionSpace(cparams,*f_d, *j_d, lambda);
        }
    }
}


template< class DataTypes >
void ConstraintCorrection< DataTypes >::addConstraintForceInMotionSpace(const core::ConstraintParams*, Data< VecDeriv > &f, const Data< MatrixDeriv>& j, const linearalgebra::BaseVector *lambda)
{
    auto force = sofa::helper::getWriteAccessor(f);

    const size_t numDOFs = mstate->getSize();
    const size_t fPrevSize = force.size();

    if (numDOFs > fPrevSize)
    {
        force.resize(numDOFs);
        for (size_t i = fPrevSize; i < numDOFs; ++i)
            force[i] = Deriv();
    }

    const MatrixDeriv& c = j.getValue();

    // force += J^T * lambda
    c.multTransposeBaseVector(force, lambda);
}

} // namespace sofa::core::behavior
