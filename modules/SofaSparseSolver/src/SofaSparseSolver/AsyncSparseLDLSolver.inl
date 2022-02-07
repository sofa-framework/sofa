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
    if (isAsyncFactorizationFinished() || !m_asyncResult.valid())
    {
        sofa::helper::ScopedAdvancedTimer setSystemMBKMatrixTimer("setSystemMBKMatrix");
        Inherit1::setSystemMBKMatrix(mparams);
        m_hasUpdatedMatrix = true;
    }
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::solveSystem()
{
    sofa::helper::ScopedAdvancedTimer invertDataCopyTimer("AsyncSolve");

    if (newInvertDataReady)
    {
        copyAsyncInvertData();
    }

    if (this->linearSystem.needInvert)
    {
        this->getMatrixInvertData(this->linearSystem.systemMatrix);
        launchAsyncFactorization();
        this->linearSystem.needInvert = false;
    }

    if (waitForAsyncTask)
    {
        waitForAsyncTask = false;
        if (m_asyncResult.valid())
            m_asyncResult.get();
    }

    if (newInvertDataReady)
    {
        copyAsyncInvertData();
    }

    this->solve(*this->linearSystem.systemMatrix, *this->linearSystem.systemLHVector, *this->linearSystem.systemRHVector);
    if (!this->linearSystem.solutionVecId.isNull())
    {
        this->executeVisitor(simulation::mechanicalvisitor::MechanicalMultiVectorFromBaseVectorVisitor(core::execparams::defaultInstance(), this->linearSystem.solutionVecId, this->linearSystem.systemLHVector, &(this->linearSystem.matrixAccessor)) );
    }
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::invert(TMatrix& M)
{
    if (this->f_saveMatrixToFile.getValue())
    {
        saveMatrix(M);
    }

    Inherit1::factorize(M, &m_asyncInvertData);
}

template <class TMatrix, class TVector, class TThreadManager>
bool AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::addJMInvJtLocal(TMatrix* M, ResMatrixType* result,
    const JMatrixType* J, SReal fact)
{
    if (newInvertDataReady)
    {
        copyAsyncInvertData();
    }
    return Inherit1::addJMInvJtLocal(M, result, J, fact);
}

template <class TMatrix, class TVector, class TThreadManager>
bool AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::hasUpdatedMatrix()
{
    return m_hasUpdatedMatrix;
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::updateSystemMatrix()
{
    m_hasUpdatedMatrix = false;
}

template <class TMatrix, class TVector, class TThreadManager>
AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::~AsyncSparseLDLSolver()
{
    if (m_asyncResult.valid())
        m_asyncResult.get();
}

template <class TMatrix, class TVector, class TThreadManager>
bool AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::isAsyncFactorizationFinished() const
{
    return m_asyncResult.valid() &&
        m_asyncResult.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::launchAsyncFactorization()
{
    m_asyncResult = std::async(std::launch::async, &AsyncSparseLDLSolver::asyncFactorization, this);
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::asyncFactorization()
{
    newInvertDataReady = false;
    this->invert(*this->linearSystem.systemMatrix);
    newInvertDataReady = true;
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::copyAsyncInvertData()
{
    if (this->invertData)
    {
        sofa::helper::ScopedAdvancedTimer invertDataCopyTimer("invertDataCopy");
        *static_cast<InvertData*>(this->invertData.get()) = m_asyncInvertData;
    }
    newInvertDataReady = false;
}
}
