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

    std::visit([&bMap, &xMap](auto&& solver)
    {
        xMap = solver.solve(bMap);
    }, m_solver);
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
        std::visit([this](auto&& solver)
        {
            solver.analyzePattern(*m_map);
        }, m_solver);

        MfilteredrowBegin = Mfiltered.rowBegin;
        MfilteredcolsIndex = Mfiltered.colsIndex;
    }

    {
        SCOPED_TIMER_VARNAME(factorizeTimer, "factorization");
        std::visit([this](auto&& solver)
        {
            solver.factorize(*m_map);
        }, m_solver);
    }

    msg_error_when(getSolverInfo() == Eigen::ComputationInfo::InvalidInput) << "Solver cannot factorize: invalid input";
    msg_error_when(getSolverInfo() == Eigen::ComputationInfo::NoConvergence) << "Solver cannot factorize: no convergence";
    msg_error_when(getSolverInfo() == Eigen::ComputationInfo::NumericalIssue) << "Solver cannot factorize: numerical issue";
}

template <class TBlockType, class EigenSolver>
Eigen::ComputationInfo EigenDirectSparseSolver<TBlockType, EigenSolver>
::getSolverInfo() const
{
    Eigen::ComputationInfo info;
    std::visit([&info](auto&& solver)
    {
        info = solver.info();
    }, m_solver);
    return info;
}

template <class TBlockType, class EigenSolver>
void EigenDirectSparseSolver<TBlockType, EigenSolver>::updateSolverOderingMethod()
{
    if (m_selectedOrderingMethod != d_orderingMethod.getValue().getSelectedId())
    {
        switch(d_orderingMethod.getValue().getSelectedId())
        {
        case 0:  m_solver.template emplace<std::variant_alternative_t<0, decltype(m_solver)> >(); break;
        case 1:  m_solver.template emplace<std::variant_alternative_t<1, decltype(m_solver)> >(); break;
        case 2:  m_solver.template emplace<std::variant_alternative_t<2, decltype(m_solver)> >(); break;
        case 3:  m_solver.template emplace<std::variant_alternative_t<3, decltype(m_solver)> >(); break;
        default: m_solver.template emplace<std::variant_alternative_t<s_defaultOrderingMethod, decltype(m_solver)> >(); break;
        }
        m_selectedOrderingMethod = d_orderingMethod.getValue().getSelectedId();
        if (m_selectedOrderingMethod >= std::variant_size_v<decltype(m_solver)>)
            m_selectedOrderingMethod = s_defaultOrderingMethod;

        MfilteredrowBegin.clear();
        MfilteredcolsIndex.clear();
        m_map.reset();
    }
}

template <class TBlockType, class EigenSolver>
EigenDirectSparseSolver<TBlockType, EigenSolver>::EigenDirectSparseSolver()
    : Inherit1()
    , d_orderingMethod(initData(&d_orderingMethod, "ordering", "Ordering method"))
{
    sofa::helper::OptionsGroup d_orderingMethodOptions{"Natural", "AMD", "COLAMD", "Metis"};

    d_orderingMethodOptions.setSelectedItem(s_defaultOrderingMethod);
    d_orderingMethod.setValue(d_orderingMethodOptions);
}

}
