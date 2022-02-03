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
#include <SofaSparseSolver/AsyncSparseLDLSolver.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorFromBaseVectorVisitor.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::linearsolver
{
template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::init()
{
    Inherit1::init();
    waitForAsyncTask = true;
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    if (isAsyncTaskFinished() || !m_asyncResult.valid())
    {
        sofa::helper::ScopedAdvancedTimer setSystemMBKMatrixTimer("setSystemMBKMatrix");
        Inherit1::setSystemMBKMatrix(mparams);
        hasNewMatrix = true;
    }
    else
    {
        hasNewMatrix = false;
    }
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::solveSystem()
{
    if (hasNewMatrix)
    {
        {
            sofa::helper::ScopedAdvancedTimer matrixCopyTimer("matrixCopy");
            this->getMatrixInvertData(this->linearSystem.systemMatrix);
            copyAsyncInvertData();
        }
        launchAsyncTask();
    }

    if (waitForAsyncTask)
    {
        m_asyncTimeStepCounter = 0;
        waitForAsyncTask = false;
        if (m_asyncResult.valid())
            m_asyncResult.get();
        copyAsyncInvertData();
    }

    this->solve(*this->linearSystem.systemMatrix, *this->linearSystem.systemLHVector, *this->linearSystem.systemRHVector);
    if (!this->linearSystem.solutionVecId.isNull())
    {
        this->executeVisitor(simulation::mechanicalvisitor::MechanicalMultiVectorFromBaseVectorVisitor(core::execparams::defaultInstance(), this->linearSystem.solutionVecId, this->linearSystem.systemLHVector, &(this->linearSystem.matrixAccessor)) );
    }
    ++m_asyncTimeStepCounter;
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::invert(TMatrix& M)
{
    // It's the same implementation than the base class, except it stores inversion data in
    // an instantiation dedicated to the async task

    this->Mfiltered.copyNonZeros(M);
    this->Mfiltered.compress();

    int n = M.colSize();

    int * M_colptr = (int *) &this->Mfiltered.getRowBegin()[0];
    int * M_rowind = (int *) &this->Mfiltered.getColsIndex()[0];
    Real * M_values = (Real *) &this->Mfiltered.getColsValue()[0];

    if(M_colptr==nullptr || M_rowind==nullptr || M_values==nullptr || this->Mfiltered.getRowBegin().size() < (size_t)n )
    {
        msg_warning() << "Invalid Linear System to solve. Please insure that there is enough constraints (not rank deficient)." ;
        return ;
    }

    Inherit1::factorize(n,M_colptr,M_rowind,M_values, &m_asyncInvertData);

    ++this->numStep;
}

template <class TMatrix, class TVector, class TThreadManager>
bool AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::isAsyncTaskFinished() const
{
    return m_asyncResult.valid() &&
    m_asyncResult.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::launchAsyncTask()
{
    m_asyncTimeStepCounter = 0;
    m_asyncResult = std::async(std::launch::async, &AsyncSparseLDLSolver::asyncTask, this);
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::asyncTask()
{
    this->invert(*this->linearSystem.systemMatrix);
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::copyAsyncInvertData()
{
    if (this->invertData)
    {
        *static_cast<InvertData*>(this->invertData.get()) = m_asyncInvertData;
    }
}
}
