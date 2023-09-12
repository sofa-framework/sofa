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

#include <sofa/component/linearsystem/config.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

#include "matrixaccumulators/AssemblingMappedMatrixAccumulator.h"

namespace sofa::component::linearsystem
{

using core::matrixaccumulator::no_check_policy;

/**
 * Assembling method benefitting from the constant sparsity pattern of the linear system in order to increase the
 * performances.
 *
 * This method must not be used if the sparsity pattern changes from a time step to another.
 *
 * The method uses the first iteration to build the data structures required to accelerate the matrix assembly, based
 * on a constant sparsity pattern.
 *
 * First iteration:
 * 1) During the accumulation of matrix contributions, the local matrix stores an ordered list of matrix elements (row
 * and column) corresponding to the order of insertion. This order must not change along the simulation.
 * 2) After the matrix assembly, the global matrix is compressed so that the ordered list of matrix elements can be
 * converted into an ordered list of ids corresponding to locations in the values array of the compressed matrix.
 *
 * Second time step and after:
 * 1) The local matrices assume the order of insertion did not change. Therefore, they rely only on the ordered list of
 * ids to know where in the values array to insert the matrix contribution. The row and column ids are useless.
 */
class SOFA_COMPONENT_LINEARSYSTEM_API ConstantSparsityPatternSystem : public MatrixLinearSystem<
    sofa::linearalgebra::CompressedRowSparseMatrix<SReal>, sofa::linearalgebra::FullVector<SReal> >
{
public:
    SOFA_CLASS(ConstantSparsityPatternSystem,
        SOFA_TEMPLATE2(MatrixLinearSystem, sofa::linearalgebra::CompressedRowSparseMatrix<SReal>, sofa::linearalgebra::FullVector<SReal>));

    using Matrix = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using Vector = sofa::linearalgebra::FullVector<SReal>;

    void applyProjectiveConstraints(const core::MechanicalParams* mparams) override;
    void resizeSystem(sofa::Size n) override;
    void clearSystem() override;

    using ConstantCRSMapping = std::unordered_map<sofa::SignedIndex, std::size_t>;

    bool isConstantSparsityPatternUsedYet() const;

protected:

    void preAssembleSystem(const core::MechanicalParams* /*mparams*/) override;

    bool m_isConstantSparsityPatternUsedYet { false };
    std::unique_ptr<ConstantCRSMapping> m_constantCRSMapping;
    sofa::type::vector<ConstantCRSMapping> m_constantCRSMappingMappedMatrices;


    ConstantSparsityPatternSystem();


    template<core::matrixaccumulator::Contribution c>
    void replaceLocalMatrices(const core::MechanicalParams* mparams,
                              LocalMatrixMaps<c, Real>& matrixMaps);

    template<core::matrixaccumulator::Contribution c>
    void replaceLocalMatrixMapped(const core::MechanicalParams* mparams, LocalMatrixMaps<c, Real>& matrixMaps);

    template <core::matrixaccumulator::Contribution c>
    void replaceLocalMatricesNonMapped(const core::MechanicalParams* mparams, LocalMatrixMaps<c, Real>& matrixMaps);


    template<core::matrixaccumulator::Contribution c>
    void reinitLocalMatrices(LocalMatrixMaps<c, Real>& matrixMaps);


    static void buildHashTable(linearalgebra::CompressedRowSparseMatrix<SReal>& M, ConstantCRSMapping& mapping);

    void makeCreateDispatcher() override;

private:
    template<Contribution c>
    static std::unique_ptr<CreateMatrixDispatcher<c>> makeCreateDispatcher();
};


}
