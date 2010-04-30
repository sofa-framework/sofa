/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution

#ifndef SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_INL
#define SOFA_COMPONENT_LINEARSOLVER_WARPPRECONDITIONER_INL

#include <sofa/component/linearsolver/WarpPreconditioner.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <iostream>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class DataTypes>
WarpPreconditioner<DataTypes>::WarpPreconditioner()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , solverName(initData(&solverName, std::string(""), "solverName", "Name of the solver/preconditioner to warp"))
    , realSolver(NULL), mstate(NULL), forceField(NULL)
{
    first = true;
}


template<class DataTypes>
void WarpPreconditioner<DataTypes>::bwdInit()
{
    mstate = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    this->getContext()->get(realSolver, solverName.getValue());
    this->getContext()->get(forceField);
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
    if (rotatedLHVId.isNull())
    {
        rotatedLHVId = VecId(VecId::V_DERIV, VecId::V_FIRST_DYNAMIC_INDEX);
        mstate->vAvail(rotatedLHVId);
        mstate->vAlloc(rotatedLHVId);
        sout << "Allocated LH vector " << rotatedLHVId << sendl;
    }
    if (rotatedRHVId.isNull())
    {
        rotatedRHVId = VecId(VecId::V_DERIV, VecId::V_FIRST_DYNAMIC_INDEX);
        mstate->vAvail(rotatedRHVId);
        mstate->vAlloc(rotatedRHVId);
        sout << "Allocated RH vector " << rotatedRHVId << sendl;
    }

    if (forceField) getRotations();

    if (first && realSolver)
    {
        realSolver->setSystemMBKMatrix(mFact, bFact, kFact);
        first=false;
    }
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::resetSystem()
{
    if (first && realSolver) realSolver->resetSystem();
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::invertSystem()
{
    if (first && realSolver) realSolver->invertSystem();
}

/// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemRHVector(VecId v)
{
    systemRHVId = v;

    //if (realSolver) realSolver->setSystemRHVector(v);
}

/// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
/// This vector will be replaced by the solution of the system once solveSystem is called
template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemLHVector(VecId v)
{
    systemLHVId = v;
    //if (realSolver) realSolver->setSystemLHVector(v);
}

/// Solve the system as constructed using the previous methods
template<class DataTypes>
void WarpPreconditioner<DataTypes>::solveSystem()
{
    //std::cout << "SOLVE" << std::endl;
    if (!realSolver || !mstate) return;
    //std::cout << ">SOLVE" << std::endl;

    if (forceField)
    {
        helper::ReadAccessor <VecDeriv> rv = *mstate->getVecDeriv(systemRHVId.index);
        helper::WriteAccessor<VecDeriv> rvR = *mstate->getVecDeriv(rotatedRHVId.index);
        unsigned int size = rv.size();
        for (unsigned int i=0; i<size; ++i) rvR[i] = data.R[i].multTranspose(rv[i]);            // rotatedRH = R^t * systemRH

        realSolver->setSystemRHVector(rotatedRHVId);
        realSolver->setSystemLHVector(rotatedLHVId);
        realSolver->solveSystem();                     // rotatedLH = M^-1 * rotatedRH

        helper::WriteAccessor<VecDeriv> lv = *mstate->getVecDeriv(systemLHVId.index);
        helper::ReadAccessor <VecDeriv> lvR = *mstate->getVecDeriv(rotatedLHVId.index);
        for (unsigned int i=0; i<size; ++i) lv[i] = data.R[i] * (lvR[i]); // systemLH = R * rotatedLH
    }
    else mstate->vOp(systemLHVId, systemRHVId);     // systemLH = rotatedLH
}

template<class TDataTypes>
void WarpPreconditioner<TDataTypes>::getRotations()
{
    if (!forceField || !mstate) return;
    unsigned int size = mstate->getSize();
    data.R.resize(size);
    for (unsigned int i=0; i<size; ++i)
        forceField->getRotation(data.R[i], i);
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
