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
#include <variant>
#include <Eigen/SparseCore>

#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::linearsolver::direct
{

/**
 * Base class for all Eigen based direct sparse solvers
 */
template<class TBlockType, class EigenSolver>
class EigenDirectSparseSolver
    : public sofa::component::linearsolver::MatrixLinearSolver<
        sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType>,
        sofa::linearalgebra::FullVector<typename sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType>::Real> >
{
public:
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType> Matrix;
    using Real = typename Matrix::Real;
    typedef sofa::linearalgebra::FullVector<Real> Vector;

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenDirectSparseSolver, TBlockType, EigenSolver),
        SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver, Matrix, Vector));

    using NaturalOrderSolver = typename EigenSolver::NaturalOrderSolver;
    using AMDOrderSolver     = typename EigenSolver::AMDOrderSolver;
    using COLAMDOrderSolver  = typename EigenSolver::COLAMDOrderSolver;
    using MetisOrderSolver   = typename EigenSolver::MetisOrderSolver;

    ~EigenDirectSparseSolver() override = default;

    void doBaseObjectInit() override;
    void reinit() override;

    using EigenSparseMatrix    = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
    using EigenSparseMatrixMap = Eigen::Map<EigenSparseMatrix>;
    using EigenVectorXdMap     = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1> >;

    void solve (Matrix& A, Vector& x, Vector& b) override;
    void invert(Matrix& A) override;

protected:

    sofa::core::objectmodel::Data<sofa::helper::OptionsGroup> d_orderingMethod;
    unsigned int m_selectedOrderingMethod { std::numeric_limits<unsigned int>::max() };

    std::variant<NaturalOrderSolver, AMDOrderSolver, COLAMDOrderSolver, MetisOrderSolver> m_solver;

    Eigen::ComputationInfo getSolverInfo() const;
    void updateSolverOderingMethod();

    sofa::linearalgebra::CompressedRowSparseMatrix<Real> Mfiltered;
    std::unique_ptr<EigenSparseMatrixMap> m_map;

    typename sofa::linearalgebra::CompressedRowSparseMatrix<Real>::VecIndex MfilteredrowBegin;
    typename sofa::linearalgebra::CompressedRowSparseMatrix<Real>::VecIndex MfilteredcolsIndex;

    EigenDirectSparseSolver();

    static constexpr unsigned int s_defaultOrderingMethod { 1 };
};

}
