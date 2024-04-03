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
#include <sofa/component/linearsystem/MatrixProjectionMethod.h>
#include <sofa/simulation/ParallelSparseMatrixProduct.h>

namespace sofa::component::linearsystem
{

/**
 * Matrix prjection method computing the matrix projection taking advantage of the constant sparsity pattern
 */
template<class TMatrix>
class ConstantSparsityProjectionMethod : public MatrixProjectionMethod<TMatrix>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ConstantSparsityProjectionMethod, TMatrix), SOFA_TEMPLATE(MatrixProjectionMethod, TMatrix));
    using PairMechanicalStates = typename BaseMatrixProjectionMethod<TMatrix>::PairMechanicalStates;
    using Block = typename Inherit1::Block;

    ConstantSparsityProjectionMethod();
    ~ConstantSparsityProjectionMethod() override;

    Data<bool> d_parallelProduct;

    void init() override;
    void reinit() override;

protected:
    void computeProjection(
        const Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor> > KMap,
        const sofa::type::fixed_array<std::shared_ptr<TMatrix>, 2> J,
        Eigen::SparseMatrix<Block, Eigen::RowMajor>& JT_K_J) override;

    using K_Type = Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor> >;
    using J_Type = Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor> >;
    using KJ_Type = Eigen::SparseMatrix<Block, Eigen::ColMajor>;
    using JT_Type = const Eigen::Transpose<const Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor> > >;
    using JTKJ_Type = Eigen::SparseMatrix<Block, Eigen::RowMajor>;

    std::unique_ptr<linearalgebra::SparseMatrixProduct< K_Type, J_Type, KJ_Type> > m_matrixProductKJ;
    std::unique_ptr<linearalgebra::SparseMatrixProduct< JT_Type, KJ_Type, JTKJ_Type> > m_matrixProductJTKJ;
};

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_CONSTANTSPARSITYPROJECTIONMETHOD_CPP)
extern template class SOFA_COMPONENT_LINEARSYSTEM_API ConstantSparsityProjectionMethod<sofa::linearalgebra::CompressedRowSparseMatrix<SReal> >;
#endif

}

