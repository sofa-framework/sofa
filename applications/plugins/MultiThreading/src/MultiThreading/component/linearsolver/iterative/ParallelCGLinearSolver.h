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

#include <MultiThreading/TaskSchedulerUser.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>

namespace multithreading::component::linearsolver::iterative
{

template<class TMatrix, class TVector>
class ParallelCGLinearSolver :
    public sofa::component::linearsolver::iterative::CGLinearSolver<TMatrix, TVector>,
    public TaskSchedulerUser
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(ParallelCGLinearSolver,TMatrix,TVector),
               SOFA_TEMPLATE2(sofa::component::linearsolver::iterative::CGLinearSolver,TMatrix,TVector),
               TaskSchedulerUser);

    using Matrix = TMatrix;
    using Vector = TVector;

    void init() override;

    void solve(Matrix& A, Vector& x, Vector& b) override;
};

#if !defined(SOFA_MULTITHREADING_PARALLELCGLINEARSOLVER_CPP)
extern template class SOFA_MULTITHREADING_PLUGIN_API
ParallelCGLinearSolver< ParallelCompressedRowSparseMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> >;
ParallelCGLinearSolver< ParallelCompressedRowSparseMatrix<sofa::type::Mat<3, 3, Real> >, sofa::linearalgebra::FullVector<SReal> >;
#endif

}
