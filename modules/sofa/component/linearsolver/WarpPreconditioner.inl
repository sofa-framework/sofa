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
#include <sofa/core/behavior/LinearSolver.h>
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
}


template<class DataTypes>
void WarpPreconditioner<DataTypes>::bwdInit()
{
    mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    this->getContext()->get(realSolver, solverName.getValue());
    this->getContext()->get(forceField);
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams)
{
    if (rotatedLHVId.isNull())
    {
        rotatedLHVId = core::VecDerivId(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
        mstate->vAvail(rotatedLHVId);
        mstate->vAlloc(rotatedLHVId);
        sout << "Allocated LH vector " << rotatedLHVId << sendl;
    }
    if (rotatedRHVId.isNull())
    {
        rotatedRHVId = core::VecDerivId(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
        mstate->vAvail(rotatedRHVId);
        mstate->vAlloc(rotatedRHVId);
        sout << "Allocated RH vector " << rotatedRHVId << sendl;
    }

    getRotations(Rinv);
    realSolver->setSystemMBKMatrix(mparams);
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::resetSystem()
{
    if (realSolver) realSolver->resetSystem();
}

template<class DataTypes>
void WarpPreconditioner<DataTypes>::invertSystem()
{
    if (realSolver) realSolver->invertSystem();
}

/// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemRHVector(core::MultiVecDerivId v)
{
    systemRHVId = v;

    //if (realSolver) realSolver->setSystemRHVector(v);
}

/// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
/// This vector will be replaced by the solution of the system once solveSystem is called
template<class DataTypes>
void WarpPreconditioner<DataTypes>::setSystemLHVector(core::MultiVecDerivId v)
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

    unsigned int size = mstate->getSize();
    if (this->frozen)
    {
        getRotations(Rcurr);
        Real tmp[9];
        //Rcurr * Rinv_i
        for (unsigned i=0; i<size*9; i+=9)
        {
            tmp[0] = Rinv[i+0] * Rcurr[i+0] + Rinv[i+3] * Rcurr[i+3] + Rinv[i+6] * Rcurr[i+6];
            tmp[1] = Rinv[i+0] * Rcurr[i+1] + Rinv[i+3] * Rcurr[i+4] + Rinv[i+6] * Rcurr[i+7];
            tmp[2] = Rinv[i+0] * Rcurr[i+2] + Rinv[i+3] * Rcurr[i+5] + Rinv[i+6] * Rcurr[i+8];

            tmp[3] = Rinv[i+1] * Rcurr[i+0] + Rinv[i+4] * Rcurr[i+3] + Rinv[i+7] * Rcurr[i+6];
            tmp[4] = Rinv[i+1] * Rcurr[i+1] + Rinv[i+4] * Rcurr[i+4] + Rinv[i+7] * Rcurr[i+7];
            tmp[5] = Rinv[i+1] * Rcurr[i+2] + Rinv[i+4] * Rcurr[i+5] + Rinv[i+7] * Rcurr[i+8];

            tmp[6] = Rinv[i+2] * Rcurr[i+0] + Rinv[i+5] * Rcurr[i+3] + Rinv[i+8] * Rcurr[i+6];
            tmp[7] = Rinv[i+2] * Rcurr[i+1] + Rinv[i+5] * Rcurr[i+4] + Rinv[i+8] * Rcurr[i+7];
            tmp[8] = Rinv[i+2] * Rcurr[i+2] + Rinv[i+5] * Rcurr[i+5] + Rinv[i+8] * Rcurr[i+8];

            Rcurr[i+0] = tmp[0]; Rcurr[i+1] = tmp[1]; Rcurr[i+2] = tmp[2];
            Rcurr[i+3] = tmp[3]; Rcurr[i+4] = tmp[4]; Rcurr[i+5] = tmp[5];
            Rcurr[i+6] = tmp[6]; Rcurr[i+7] = tmp[7]; Rcurr[i+8] = tmp[8];
        }
        //int element = 50;printf("%f %f %f\n%f %f %f\n%f %f %f\n----------------------\n",Rcurr[element*9],Rcurr[element*9+1],Rcurr[element*9+2],Rcurr[element*9+3],Rcurr[element*9+4],Rcurr[element*9+5],Rcurr[element*9+6],Rcurr[element*9+7],Rcurr[element*9+8]);
    }

    if (forceField)
    {
        //Solve lv = R * M-1 * R^t * rv

        //<TO REMOVE>
        //helper::ReadAccessor <VecDeriv> rv = *mstate->getVecDeriv(systemRHVId.index);
        //helper::WriteAccessor<VecDeriv> rvR = *mstate->getVecDeriv(rotatedRHVId.index);
        const Data<VecDeriv>* dataRv = mstate->read(systemRHVId.getId(mstate));
        const VecDeriv& rv = dataRv->getValue();
        Data<VecDeriv>* dataRvR = mstate->write(rotatedRHVId);
        VecDeriv& rvR = *dataRvR->beginEdit();

        //Solve rvR = Rcur^t * rv
        unsigned int k = 0,l = 0;
        while (l < size)
        {
            rvR[l][0] = Rcurr[k + 0] * rv[l][0] + Rcurr[k + 3] * rv[l][1] + Rcurr[k + 6] * rv[l][2];
            rvR[l][1] = Rcurr[k + 1] * rv[l][0] + Rcurr[k + 4] * rv[l][1] + Rcurr[k + 7] * rv[l][2];
            rvR[l][2] = Rcurr[k + 2] * rv[l][0] + Rcurr[k + 5] * rv[l][1] + Rcurr[k + 8] * rv[l][2];
            l++;
            k+=9;
        }

        dataRvR->endEdit();

        //Solve lvR = M-1 * rvR
        realSolver->setSystemRHVector(systemRHVId);
        realSolver->setSystemLHVector(rotatedLHVId);
        realSolver->solveSystem();

        //<TO REMOVE>
        //helper::WriteAccessor<VecDeriv> lv = *mstate->getVecDeriv(systemLHVId.index);
        //helper::ReadAccessor <VecDeriv> lvR = *mstate->getVecDeriv(rotatedLHVId.index);

        Data<VecDeriv>* dataLv = mstate->write(systemLHVId.getId(mstate));
        VecDeriv& lv = *dataLv->beginEdit();
        const Data<VecDeriv>* dataLvR = mstate->read(rotatedRHVId);
        const VecDeriv& lvR = dataLvR->getValue();

        //Solve lv = R * lvR
        k = 0; l = 0;
        while (l < size)
        {
            lv[l][0] = Rcurr[k + 0] * lvR[l][0] + Rcurr[k + 1] * lvR[l][1] + Rcurr[k + 2] * lvR[l][2];
            lv[l][1] = Rcurr[k + 3] * lvR[l][0] + Rcurr[k + 4] * lvR[l][1] + Rcurr[k + 5] * lvR[l][2];
            lv[l][2] = Rcurr[k + 6] * lvR[l][0] + Rcurr[k + 7] * lvR[l][1] + Rcurr[k + 8] * lvR[l][2];
            l++;
            k+=9;
        }
        dataLv->endEdit();

    }
    else mstate->vOp(systemLHVId.getId(mstate), systemRHVId.getId(mstate));     // systemLH = rotatedLH
}

template<class TDataTypes>
void WarpPreconditioner<TDataTypes>::getRotations(TBaseVector & R)
{
    if (!mstate) return;
    unsigned int size = mstate->getSize();
    R.resize(size*9);

    if (forceField != NULL)
    {
        Transformation Rotation;
        for(unsigned int k = 0; k < size; k++)
        {
            forceField->getRotation(Rotation, k);
            for (int j=0; j<3; j++)
            {
                for (int i=0; i<3; i++)
                {
                    R[k*9+j*3+i] = (Real)Rotation[j][i];
                }
            }
        }
    }
    else
    {
        serr << "No rotation defined : use Identity !!";
        for(unsigned int k = 0; k < size; k++)
        {
            R[k*9] = R[k*9+4] = R[k*9+8] = 1.0f;
            R[k*9+1] = R[k*9+2] = R[k*9+3] = R[k*9+5] = R[k*9+6] = R[k*9+7] = 0.0f;
        }
    }
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
