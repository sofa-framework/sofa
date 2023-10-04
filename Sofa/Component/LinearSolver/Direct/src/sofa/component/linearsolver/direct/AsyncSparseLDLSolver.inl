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

#include <sofa/component/linearsolver/direct/AsyncSparseLDLSolver.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalMultiVectorFromBaseVectorVisitor.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::linearsolver::direct
{

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::init()
{
    Inherit1::init();
    waitForAsyncTask = true;
    m_asyncThreadInvertData = &m_secondInvertData;
    m_mainThreadInvertData = static_cast<InvertData*>(this->invertData.get());
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    if (isAsyncFactorizationFinished() || !m_asyncResult.valid())
    {
        SCOPED_TIMER_VARNAME(setSystemMBKMatrixTimer, "setSystemMBKMatrix");
        Inherit1::setSystemMBKMatrix(mparams);
        m_hasUpdatedMatrix = true;
    }
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::solveSystem()
{
    SCOPED_TIMER_VARNAME(invertDataCopyTimer, "AsyncSolve");

    if (newInvertDataReady)
    {
        swapInvertData();
    }

    if (this->linearSystem.needInvert)
    {
        if (this->invertData == nullptr)
        {
            this->getMatrixInvertData(this->getSystemMatrix());
            m_mainThreadInvertData = static_cast<InvertData*>(this->invertData.get());
        }
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
        swapInvertData();
    }

    this->solve(*this->getSystemMatrix(), *this->getSystemLHVector(), *this->getSystemRHVector());

    if (!this->linearSystem.solutionVecId.isNull())
    {
        if (this->l_linearSystem)
        {
            this->l_linearSystem->dispatchSystemSolution(this->linearSystem.solutionVecId);
        }
    }
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::solve(Matrix& M, Vector& x, Vector& b)
{
    SOFA_UNUSED(M);

    Inherit1::solve_cpu(&x[0],&b[0], m_mainThreadInvertData);
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::invert(TMatrix& M)
{
    Inherit1::factorize(M, m_asyncThreadInvertData);
}

template <class TMatrix, class TVector, class TThreadManager>
bool AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::addJMInvJtLocal(TMatrix* M, ResMatrixType* result,
    const JMatrixType* J, SReal fact)
{
    SOFA_UNUSED(M);

    if (newInvertDataReady)
    {
        swapInvertData();
    }
    return Inherit1::doAddJMInvJtLocal(result, J, fact, m_mainThreadInvertData);
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
    this->invert(*this->getSystemMatrix());
    newInvertDataReady = true;
}

template <class TMatrix, class TVector, class TThreadManager>
void AsyncSparseLDLSolver<TMatrix, TVector, TThreadManager>::swapInvertData()
{
    if (this->invertData)
    {
        std::swap(m_mainThreadInvertData, m_asyncThreadInvertData);
    }
    newInvertDataReady = false;
}

} // namespace sofa::component::linearsolver::direct
