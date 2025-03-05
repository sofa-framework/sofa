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
#include <sofa/component/linearsystem/BaseMatrixProjectionMethod.h>
#include <Eigen/Sparse>
#include <optional>

namespace sofa::component::linearsystem
{

/**
 * The component is a @MatrixMapping computing the matrix projection using
 * the Eigen library.
 */
template<class TMatrix>
class MatrixProjectionMethod : public BaseMatrixProjectionMethod<TMatrix>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MatrixProjectionMethod, TMatrix), SOFA_TEMPLATE(BaseMatrixProjectionMethod, TMatrix));
    using PairMechanicalStates = typename BaseMatrixProjectionMethod<TMatrix>::PairMechanicalStates;
    using Block = typename TMatrix::Block;

    ~MatrixProjectionMethod() override;

    void computeMatrixJacobians(const core::MechanicalParams* mparams,
                                const MappingGraph& mappingGraph,
                                TMatrix* matrixToProject);

protected:
    explicit MatrixProjectionMethod(const PairMechanicalStates& states);
    MatrixProjectionMethod();

    void reinit() override;

    virtual void computeMatrixProduct(const MappingGraph& mappingGraph,
                          TMatrix* matrixToProject,
                          linearalgebra::BaseMatrix* globalMatrix);

    virtual void projectMatrixToGlobalMatrix(const core::MechanicalParams* mparams,
                                     const MappingGraph& mappingGraph,
                                     TMatrix* matrixToProject,
                                     linearalgebra::BaseMatrix* globalMatrix) override;

    /// Given a Mechanical State and its matrix, identifies the nodes affected by the matrix
    std::vector<unsigned int> identifyAffectedDoFs(BaseMechanicalState* mstate, TMatrix* crs);

    /**
     * Build the jacobian matrices of mappings from a mapped state to its top most parents (in the
     * sense of mappings)
     */
    MappingJacobians<TMatrix> computeJacobiansFrom(BaseMechanicalState* mstate, const core::MechanicalParams* mparams, const MappingGraph& mappingGraph, TMatrix* crs);

    core::objectmodel::BaseContext* getSolveContext();

    /**
     * Add the local matrix which has been built locally to the main global matrix, using the Eigen library
     *
     * @remark Eigen manages the matrix operations better than CompressedRowSparseMatrix. In terms of performances, it is
     * preferable to go with Eigen.
     *
     * @param mstatePair The mapped mechanical state which the local matrix is associated
     * @param mappedMatrix The local matrix
     * @param jacobians The required mapping jacobians to project from a mechanical state toward the top most mechanical states
     * @param mappingGraph The mapping graph used to know the relationships between mechanical states. In particular, it
     * is used to know where in the global matrix the local matrix must be added.
     * @param globalMatrix Matrix in which the local matrix is added.
     */
    void addMappedMatrixToGlobalMatrixEigen(
        sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2> mstatePair,
        TMatrix* mappedMatrix,
        sofa::type::fixed_array<MappingJacobians<TMatrix>, 2> jacobians,
        const MappingGraph& mappingGraph,
        linearalgebra::BaseMatrix* globalMatrix);

    Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor> > makeEigenMap(const TMatrix& matrix);

    virtual void computeProjection(
        const Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor> > KMap,
        const sofa::type::fixed_array<std::shared_ptr<TMatrix>, 2> J,
        Eigen::SparseMatrix<Block, Eigen::RowMajor>& JT_K_J);

    Data<bool> d_areJacobiansConstant; ///< True if mapping jacobians are considered constant over time. They are computed only the first time.

    std::optional<sofa::type::fixed_array<MappingJacobians<TMatrix>, 2>> m_mappingJacobians;
};

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_EIGENMATRIXMAPPING_CPP)
extern template class SOFA_COMPONENT_LINEARSYSTEM_API MatrixProjectionMethod<linearalgebra::CompressedRowSparseMatrix<SReal> >;
#endif

}
