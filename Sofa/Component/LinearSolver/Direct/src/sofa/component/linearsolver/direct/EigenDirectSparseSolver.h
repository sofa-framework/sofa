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
#include <sofa/component/linearsolver/direct/config.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/component/linearsolver/ordering/OrderingMethodAccessor.h>
#include <variant>
#include <Eigen/SparseCore>
#include <sofa/component/linearsolver/direct/EigenSolverFactory.h>

#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::linearsolver::direct
{

/**
 * Base class for all Eigen based direct sparse solvers
 */
template<class TBlockType, class TEigenSolverFactory>
class EigenDirectSparseSolver
    : public ordering::OrderingMethodAccessor<
        sofa::component::linearsolver::MatrixLinearSolver<
            sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType>,
            sofa::linearalgebra::FullVector<typename sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType>::Real>
        >
    >
{
public:
    using Matrix = sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType>;
    using Real = typename Matrix::Real;
    using Vector = sofa::linearalgebra::FullVector<Real>;

    using EigenSolverFactory = TEigenSolverFactory;

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenDirectSparseSolver, TBlockType, EigenSolverFactory),
        SOFA_TEMPLATE(ordering::OrderingMethodAccessor, SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, Matrix, Vector)));

    ~EigenDirectSparseSolver() override = default;

    void init() override;
    void reinit() override;

    using EigenSparseMatrix    = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
    using EigenSparseMatrixMap = Eigen::Map<EigenSparseMatrix>;
    using EigenVectorXdMap     = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1> >;

    void solve (Matrix& A, Vector& x, Vector& b) override;
    void invert(Matrix& A) override;

protected:

    DeprecatedAndRemoved d_orderingMethod;
    std::string m_selectedOrderingMethod;

    std::unique_ptr<BaseEigenSolverProxy> m_solver;

    [[nodiscard]] Eigen::ComputationInfo getSolverInfo() const;
    void updateSolverOderingMethod();

    sofa::linearalgebra::CompressedRowSparseMatrix<Real> Mfiltered;
    std::unique_ptr<EigenSparseMatrixMap> m_map;

    typename sofa::linearalgebra::CompressedRowSparseMatrix<Real>::VecIndex MfilteredrowBegin;
    typename sofa::linearalgebra::CompressedRowSparseMatrix<Real>::VecIndex MfilteredcolsIndex;

    static constexpr unsigned int s_defaultOrderingMethod { 1 };
};

}
