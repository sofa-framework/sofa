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
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.h>
#include <sofa/core/ComponentLibrary.h>

#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::linearsolver::direct
{
template <class TBlockType, class EigenSolver>
void EigenDirectSparseSolver<TBlockType, EigenSolver>
    ::init()
{
    Inherit1::init();
    updateSolverOderingMethod();
}

template <class TBlockType, class EigenSolver>
void EigenDirectSparseSolver<TBlockType, EigenSolver>
    ::reinit()
{
    updateSolverOderingMethod();
}

template <class TBlockType, class EigenSolver>
void EigenDirectSparseSolver<TBlockType, EigenSolver>
    ::solve(Matrix& A, Vector& x, Vector& b)
{
    SOFA_UNUSED(A);

    EigenVectorXdMap xMap(x.ptr(), x.size());
    EigenVectorXdMap bMap(b.ptr(), b.size());

    m_solver->solve(bMap, xMap);
}

template <class TBlockType, class EigenSolver>
void EigenDirectSparseSolver<TBlockType, EigenSolver>
    ::invert(Matrix& A)
{
    {
        SCOPED_TIMER_VARNAME(copyTimer, "copyMatrixData");
        Mfiltered.copyNonZeros(A);
        Mfiltered.compress();
    }

    m_map = std::make_unique<EigenSparseMatrixMap>(Mfiltered.rows(), Mfiltered.cols(), Mfiltered.getColsValue().size(),
                                                   (typename EigenSparseMatrixMap::StorageIndex*)Mfiltered.rowBegin.data(),
                                                   (typename EigenSparseMatrixMap::StorageIndex*)Mfiltered.colsIndex.data(),
                                                   Mfiltered.colsValue.data());

    const bool analyzePattern = (MfilteredrowBegin != Mfiltered.rowBegin) || (MfilteredcolsIndex != Mfiltered.colsIndex);

    if (analyzePattern)
    {
        SCOPED_TIMER_VARNAME(patternAnalysisTimer, "patternAnalysis");
        m_solver->analyzePattern(*m_map);

        MfilteredrowBegin = Mfiltered.rowBegin;
        MfilteredcolsIndex = Mfiltered.colsIndex;
    }

    {
        SCOPED_TIMER_VARNAME(factorizeTimer, "factorization");
        m_solver->factorize(*m_map);
    }

    msg_error_when(getSolverInfo() == Eigen::ComputationInfo::InvalidInput) << "Solver cannot factorize: invalid input";
    msg_error_when(getSolverInfo() == Eigen::ComputationInfo::NoConvergence) << "Solver cannot factorize: no convergence";
    msg_error_when(getSolverInfo() == Eigen::ComputationInfo::NumericalIssue) << "Solver cannot factorize: numerical issue";
}

template <class TBlockType, class EigenSolver>
Eigen::ComputationInfo EigenDirectSparseSolver<TBlockType, EigenSolver>
::getSolverInfo() const
{
    return m_solver->info();
}

template <class TBlockType, class EigenSolver>
void EigenDirectSparseSolver<TBlockType, EigenSolver>::updateSolverOderingMethod()
{
    if (this->l_orderingMethod)
    {
        if (m_selectedOrderingMethod != this->l_orderingMethod->methodName())
        {
            m_selectedOrderingMethod = this->l_orderingMethod->methodName();

            if (EigenSolverFactory::template hasSolver<Real>(m_selectedOrderingMethod))
            {
                m_solver = std::unique_ptr<BaseEigenSolverProxy>(EigenSolverFactory::template getSolver<Real>(m_selectedOrderingMethod));
            }
            else
            {
                std::set<std::string> listAvailableOrderingMethods;
                for (const auto& [orderingMethodName, _] : EigenSolverFactory::registeredSolvers())
                {
                    if (EigenSolverFactory::template hasSolver<Real>(orderingMethodName))
                    {
                        listAvailableOrderingMethods.insert(orderingMethodName);
                    }
                }

                msg_error() << "This solver does not support the ordering method called '"
                    << m_selectedOrderingMethod << "' found in the component "
                    << this->l_orderingMethod->getPathName() << ". The list of available methods are: "
                    << sofa::helper::join(listAvailableOrderingMethods, ",");
                this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            }

            MfilteredrowBegin.clear();
            MfilteredcolsIndex.clear();
            m_map.reset();
        }
    }
    else
    {
        msg_fatal() << "OrderingMethod missing.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

}
