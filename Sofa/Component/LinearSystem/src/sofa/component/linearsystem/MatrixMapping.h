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

#include <fstream>
#include <sofa/component/linearsystem/MappingGraph.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/StateAccessor.h>

namespace sofa::component::linearsystem
{

template<class TMatrix>
class MatrixMapping : public core::behavior::StateAccessor
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(MatrixMapping, TMatrix), core::behavior::StateAccessor);

    using PairMechanicalStates = sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2>;

    ~MatrixMapping() override;

    virtual bool hasPairStates(const PairMechanicalStates& pairStates) const;

    virtual void projectMatrixToGlobalMatrix(const core::MechanicalParams* mparams,
        const MappingGraph& mappingGraph,
        TMatrix* matrixToProject,
        linearalgebra::BaseMatrix* globalMatrix) = 0;

protected:
    explicit MatrixMapping(const PairMechanicalStates& states);
    MatrixMapping() = default;
};


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
    using MatrixMapping<TMatrix>::MatrixMapping;

    /// Given a Mechanical State and its matrix, identifies the nodes affected by the matrix
    std::vector<unsigned int> identifyAffectedDoFs(BaseMechanicalState* mstate, TMatrix* crs);

    /**
    * Build the jacobian matrices of mappings from a mapped state to its top most parents (in the
    * sense of mappings)
    */
    MappingJacobians<TMatrix> computeJacobiansFrom(BaseMechanicalState* mstate, const core::MechanicalParams* mparams, const MappingGraph& mappingGraph, TMatrix* crs);

    core::objectmodel::BaseContext* getSolveContext();
};

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_MATRIXMAPPING_CPP)
extern template class SOFA_COMPONENT_LINEARSYSTEM_API MatrixMapping<sofa::linearalgebra::CompressedRowSparseMatrix<SReal> >;
extern template class SOFA_COMPONENT_LINEARSYSTEM_API EigenMatrixMapping<sofa::linearalgebra::CompressedRowSparseMatrix<SReal> >;
#endif

} // namespace sofa::component::linearsystem
