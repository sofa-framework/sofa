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
#include <sofa/component/linearsystem/MatrixMapping.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <optional>

namespace sofa::component::linearsystem
{

template<class TMatrix>
class EigenMatrixMapping : public MatrixMapping<TMatrix>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(EigenMatrixMapping, TMatrix), SOFA_TEMPLATE(MatrixMapping, TMatrix));
    using PairMechanicalStates = typename MatrixMapping<TMatrix>::PairMechanicalStates;

    ~EigenMatrixMapping() override;

    void projectMatrixToGlobalMatrix(const core::MechanicalParams* mparams,
        const MappingGraph& mappingGraph,
        TMatrix* matrixToProject,
        linearalgebra::BaseMatrix* globalMatrix) override;

protected:
    explicit EigenMatrixMapping(const PairMechanicalStates& states);
    EigenMatrixMapping();

    /// Given a Mechanical State and its matrix, identifies the nodes affected by the matrix
    std::vector<unsigned int> identifyAffectedDoFs(BaseMechanicalState* mstate, TMatrix* crs);

    /**
    * Build the jacobian matrices of mappings from a mapped state to its top most parents (in the
    * sense of mappings)
    */
    MappingJacobians<TMatrix> computeJacobiansFrom(BaseMechanicalState* mstate, const core::MechanicalParams* mparams, const MappingGraph& mappingGraph, TMatrix* crs);

    core::objectmodel::BaseContext* getSolveContext();

    Data<bool> d_areJacobiansConstant; ///< True if mapping jacobians are considered constant over time. They are computed only the first time.

    std::optional<sofa::type::fixed_array<MappingJacobians<TMatrix>, 2>> m_mappingJacobians;
};

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_EIGENMATRIXMAPPING_CPP)
extern template class SOFA_COMPONENT_LINEARSYSTEM_API EigenMatrixMapping<linearalgebra::CompressedRowSparseMatrix<SReal> >;
#endif

}
