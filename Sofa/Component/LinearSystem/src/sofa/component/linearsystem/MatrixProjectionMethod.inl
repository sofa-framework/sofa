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
#include <sofa/component/linearsystem/BaseMatrixProjectionMethod.inl>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <Eigen/Sparse>


namespace sofa::component::linearsystem
{

template <class TMatrix>
MatrixProjectionMethod<TMatrix>::~MatrixProjectionMethod() = default;

template <class TMatrix>
MatrixProjectionMethod<TMatrix>::MatrixProjectionMethod()
    : d_areJacobiansConstant(initData(&d_areJacobiansConstant, false, "areJacobiansConstant", "True if mapping jacobians are considered constant over time. They are computed only the first time."))
{}

template <class TMatrix>
void MatrixProjectionMethod<TMatrix>::reinit()
{
    Inherit1::reinit();

    //cached jacobians are invalidated
    m_mappingJacobians.reset();
}

template <class TMatrix>
MatrixProjectionMethod<TMatrix>::MatrixProjectionMethod(
    const PairMechanicalStates& states) : MatrixProjectionMethod()
{
    this->setPairStates(states);
}

template <class BlockType>
void computeProjection(
    const Eigen::Map<Eigen::SparseMatrix<BlockType, Eigen::RowMajor> > KMap,
    const sofa::type::fixed_array<std::shared_ptr<linearalgebra::CompressedRowSparseMatrix<BlockType>>, 2> J,
    Eigen::SparseMatrix<BlockType, Eigen::RowMajor>& JT_K_J);

template <class BlockType>
void addToGlobalMatrix(linearalgebra::BaseMatrix* globalMatrix, Eigen::SparseMatrix<BlockType, Eigen::RowMajor> JT_K_J, const type::Vec2u positionInGlobalMatrix);

template <class TMatrix>
void MatrixProjectionMethod<TMatrix>::addMappedMatrixToGlobalMatrixEigen(
    sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2> mstatePair,
    TMatrix* mappedMatrix,
    sofa::type::fixed_array<MappingJacobians<TMatrix>, 2> jacobians,
    const MappingGraph& mappingGraph, linearalgebra::BaseMatrix* globalMatrix)
{
    if (!mappedMatrix)
    {
        return;
    }

    if (mappedMatrix->rows() == 0 || mappedMatrix->cols() == 0)
    {
        return;
    }

    if (mappedMatrix->getColsValue().empty() && mappedMatrix->btemp.empty())
    {
        return;
    }

    mappedMatrix->fullRows();

    const auto KMap = makeEigenMap(*mappedMatrix);

    // nb rows of K = size of first mechanical state
    msg_error_when(sofa::Size(mappedMatrix->rows()) != mstatePair[0]->getMatrixSize(), "MatrixMapping")
        << "[K] Incompatible matrix size [rows] " << mappedMatrix->rows() << " " << mstatePair[0]->getMatrixSize();
    // nb cols of K = size of second mechanical state
    msg_error_when(sofa::Size(mappedMatrix->cols()) != mstatePair[1]->getMatrixSize(), "MatrixMapping")
        << "[K] Incompatible matrix size [cols] " << mappedMatrix->cols() << " " << mstatePair[1]->getMatrixSize();

    const auto inputs1 = mappingGraph.getTopMostMechanicalStates(mstatePair[0]);
    const auto inputs2 = mappingGraph.getTopMostMechanicalStates(mstatePair[1]);

    std::set<core::behavior::BaseMechanicalState*> inputs;
    inputs.insert(inputs1.begin(), inputs1.end());
    inputs.insert(inputs2.begin(), inputs2.end());

    std::set< std::pair<core::behavior::BaseMechanicalState*, core::behavior::BaseMechanicalState*> > uniquePairs;
    for (auto* a : inputs)
    {
        for (auto* b : inputs)
        {
            uniquePairs.insert({a, b});
        }
    }

    for (const auto& [a, b] : uniquePairs)
    {
        const sofa::type::fixed_array<std::shared_ptr<TMatrix>, 2> J
        { jacobians[0].getJacobianFrom(a), jacobians[1].getJacobianFrom(b) };

        if (J[0])
        {
            // nb rows of J[0] = size of first mechanical state
            msg_error_when(sofa::Size(J[0]->rows()) != mstatePair[0]->getMatrixSize(), "MatrixMapping")
                    << "[J0] Incompatible matrix size [rows] " << J[0]->rows() << " " << mstatePair[0]->getMatrixSize();
            msg_error_when(sofa::Size(J[0]->cols()) != a->BaseMechanicalState::getMatrixSize(), "MatrixMapping")
                    << "[J0] Incompatible matrix size [cols] " << J[0]->cols() << " " << a->BaseMechanicalState::getMatrixSize();
        }

        if (J[1])
        {
            // nb rows of J[1] = size of second mechanical state
            msg_error_when(sofa::Size(J[1]->rows()) != mstatePair[1]->getMatrixSize(), "MatrixMapping")
                    << "[J1] Incompatible matrix size [rows] " << J[1]->rows() << " " << mstatePair[1]->getMatrixSize();
            msg_error_when(sofa::Size(J[1]->cols()) != b->BaseMechanicalState::getMatrixSize(), "MatrixMapping")
                    << "[J1] Incompatible matrix size [cols] " << J[1]->cols() << " " << b->BaseMechanicalState::getMatrixSize();
        }

        Eigen::SparseMatrix<Block, Eigen::RowMajor> JT_K_J;
        computeProjection(KMap, J, JT_K_J);

        const type::Vec2u positionInGlobalMatrix = mappingGraph.getPositionInGlobalMatrix(a, b);

        addToGlobalMatrix<Block>(globalMatrix, JT_K_J, positionInGlobalMatrix);
    }
}

template <class TMatrix>
Eigen::Map<Eigen::SparseMatrix<typename MatrixProjectionMethod<TMatrix>::Block,
Eigen::RowMajor>> MatrixProjectionMethod<TMatrix>::makeEigenMap(const TMatrix& matrix)
{
    using EigenMap = Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor> >;
    return EigenMap(
            static_cast<typename EigenMap::Index>(matrix.rows()),
            static_cast<typename EigenMap::Index>(matrix.cols()),
            static_cast<typename EigenMap::Index>(matrix.getColsValue().size()),
            (typename EigenMap::StorageIndex*)matrix.rowBegin.data(),
            (typename EigenMap::StorageIndex*)matrix.colsIndex.data(),
            (typename EigenMap::Scalar*)matrix.colsValue.data());
}

template <class TMatrix>
void MatrixProjectionMethod<TMatrix>::computeProjection(
    const Eigen::Map<Eigen::SparseMatrix<Block, Eigen::RowMajor>> KMap,
    const sofa::type::fixed_array<std::shared_ptr<TMatrix>, 2> J,
    Eigen::SparseMatrix<Block, Eigen::RowMajor>& JT_K_J)
{
    if (J[0] && J[1])
    {
        const auto JMap0 = makeEigenMap(*J[0]);
        const auto JMap1 = makeEigenMap(*J[1]);
        JT_K_J = JMap0.transpose() * ( KMap * JMap1);
    }
    else if (J[0] && !J[1])
    {
        const auto JMap0 = makeEigenMap(*J[0]);
        JT_K_J = JMap0.transpose() * KMap;
    }
    else if (!J[0] && J[1])
    {
        const auto JMap1 = makeEigenMap(*J[1]);
        JT_K_J = KMap * JMap1;
    }
    else
    {
        JT_K_J = KMap;
    }
}

template <class BlockType>
void addToGlobalMatrix(linearalgebra::BaseMatrix* globalMatrix, Eigen::SparseMatrix<BlockType, Eigen::RowMajor> JT_K_J, const type::Vec2u positionInGlobalMatrix)
{
    for (int k = 0; k < JT_K_J.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<BlockType, Eigen::RowMajor>::InnerIterator it(JT_K_J,k); it; ++it)
        {
            globalMatrix->add(it.row() + positionInGlobalMatrix[0], it.col() + positionInGlobalMatrix[1], it.value());
        }
    }
}

template <class TMatrix>
void MatrixProjectionMethod<TMatrix>::computeMatrixJacobians(const core::MechanicalParams* mparams, const MappingGraph& mappingGraph, TMatrix* matrixToProject)
{
    if (!m_mappingJacobians.has_value() || !d_areJacobiansConstant.getValue())
    {
        const MappingJacobians<TMatrix> J0 = computeJacobiansFrom(
            this->l_mechanicalStates[0], mparams, mappingGraph, matrixToProject);

        const MappingJacobians<TMatrix> J1 =
                (this->l_mechanicalStates[0] == this->l_mechanicalStates[1]) ?
                    J0 : computeJacobiansFrom(this->l_mechanicalStates[1], mparams, mappingGraph, matrixToProject);

        m_mappingJacobians.emplace(J0, J1);
    }
}

template <class TMatrix>
void MatrixProjectionMethod<TMatrix>::computeMatrixProduct(const MappingGraph& mappingGraph, TMatrix* matrixToProject, linearalgebra::BaseMatrix* globalMatrix)
{
    this->addMappedMatrixToGlobalMatrixEigen(
        {this->l_mechanicalStates[0], this->l_mechanicalStates[1]},
        matrixToProject, *m_mappingJacobians, mappingGraph,
        globalMatrix);
}

template <class TMatrix>
void MatrixProjectionMethod<TMatrix>::projectMatrixToGlobalMatrix(const core::MechanicalParams* mparams,
    const MappingGraph& mappingGraph,
    TMatrix* matrixToProject, linearalgebra::BaseMatrix* globalMatrix)
{
    computeMatrixJacobians(mparams, mappingGraph, matrixToProject);
    computeMatrixProduct(mappingGraph, matrixToProject, globalMatrix);
}

template <class TMatrix>
std::vector<unsigned> MatrixProjectionMethod<TMatrix>::identifyAffectedDoFs(
    BaseMechanicalState* mstate, TMatrix* crs)
{
    const auto blockSize = mstate->getMatrixBlockSize();
    std::set<unsigned int> setAffectedDoFs;

    for (std::size_t it_rows_k = 0; it_rows_k < crs->rowIndex.size(); it_rows_k++)
    {
        const auto row = crs->rowIndex[it_rows_k];
        {
            const sofa::SignedIndex dofId = row / blockSize;
            setAffectedDoFs.insert(dofId);
        }
        typename TMatrix::Range rowRange(crs->rowBegin[it_rows_k], crs->rowBegin[it_rows_k + 1]);
        for (auto xj = rowRange.begin(); xj < rowRange.end(); ++xj) // for each non-null block
        {
            const sofa::SignedIndex col = crs->colsIndex[xj];
            const sofa::SignedIndex dofId = col / blockSize;
            setAffectedDoFs.insert(dofId);
        }
    }

    return std::vector( setAffectedDoFs.begin(), setAffectedDoFs.end() );
}

template <class TMatrix>
MappingJacobians<TMatrix> MatrixProjectionMethod<TMatrix>::computeJacobiansFrom(
    BaseMechanicalState* mstate, const core::MechanicalParams* mparams,
    const MappingGraph& mappingGraph, TMatrix* crs)
{
    core::ConstraintParams cparams(*mparams);

    MappingJacobians<TMatrix> jacobians(*mstate);

    if (!mappingGraph.hasAnyMappingInput(mstate))
    {
        return jacobians;
    }

    auto mappingJacobianId = sofa::core::MatrixDerivId::mappingJacobian();

    // this clears the matrix identified by mappingJacobian() among others
    simulation::mechanicalvisitor::MechanicalResetConstraintVisitor(&cparams).execute(this->getSolveContext());

    // optimisation to build only the relevant entries of the jacobian matrices
    // The relevant entries are the ones that have an influence on the result
    // of the product J^T * K * J.
    // J does not need to be fully computed if K is sparse.
    {
        crs->compress();
        const std::vector<unsigned> listAffectedDoFs = identifyAffectedDoFs(mstate, crs);

        if (listAffectedDoFs.empty())
        {
            return jacobians;
        }
        mstate->buildIdentityBlocksInJacobian(listAffectedDoFs, mappingJacobianId);
    }

    // apply the mappings from the bottom to the top, so it builds the jacobian
    // matrices, transforming the space from the input DoFs to the space of the
    // top most DoFs
    const auto parentMappings = mappingGraph.getBottomUpMappingsFrom(mstate);
    for (auto* mapping : parentMappings)
    {
        mapping->applyJT(&cparams, mappingJacobianId, mappingJacobianId);
    }

    // copy the jacobian matrix stored in the mechanical states into a local
    // matrix data structure
    const auto inputs = mappingGraph.getTopMostMechanicalStates(mstate);
    for (auto* input : inputs)
    {
        auto J = std::make_shared<TMatrix>();
        jacobians.addJacobianToTopMostParent(J, input);
        J->resize(mstate->getMatrixSize(), input->getMatrixSize());
        unsigned int offset {};
        input->copyToBaseMatrix(J.get(), mappingJacobianId, offset);

        //set the sizes again because in some cases they are changed in copyToBaseMatrix
        J->nCol = input->getMatrixSize();
        J->nRow = mstate->getMatrixSize();
        J->nBlockCol = J->nCol;
        J->nBlockRow = J->nRow;

        J->fullRows();
    }

    return jacobians;
}

template <class TMatrix>
core::objectmodel::BaseContext* MatrixProjectionMethod<TMatrix>::getSolveContext()
{
    auto* linearSolver = this->getContext()->template get<sofa::core::behavior::LinearSolver>(core::objectmodel::BaseContext::Local);
    if (linearSolver)
    {
        return linearSolver->getContext();
    }
    linearSolver = this->getContext()->template get<sofa::core::behavior::LinearSolver>(core::objectmodel::BaseContext::SearchUp);
    if (linearSolver)
    {
        return linearSolver->getContext();
    }

    return this->getContext();
}

}
