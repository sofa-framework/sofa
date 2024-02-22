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
#include <sofa/component/linearsystem/MappingGraph.h>
#include <sofa/core/behavior/StateAccessor.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>

namespace sofa::component::linearsystem
{

/**
 * A component associated to a pair of @BaseMechanicalState able to perform
 * the projection of a matrix from the space of the @BaseMechanicalState's
 * to the main space using the mapping graph.
 *
 * Basically, if K is the matrix to be projected, it computes J, and then the
 * product J^T * K * J. The J matrix comes from the chain of mappings from
 * the @BaseMechanicalState to its top most parents.
 */
template<class TMatrix>
class BaseMatrixProjectionMethod : public core::behavior::StateAccessor
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(BaseMatrixProjectionMethod, TMatrix), core::behavior::StateAccessor);

    using PairMechanicalStates = sofa::type::fixed_array<core::behavior::BaseMechanicalState*, 2>;

    ~BaseMatrixProjectionMethod() override;

    virtual bool hasPairStates(const PairMechanicalStates& pairStates) const;
    void setPairStates(const PairMechanicalStates& pairStates);

    /**
     * \brief The main function of the component: it compute the mappings
     * jacobian matrices, then the projection of the provided matrix.
     *
     * \param mparams
     * \param mappingGraph The current mapping graph linking all the @BaseMechanicalState
     * \param matrixToProject The matrix to project. Its size must be compatible
     * with the sizes of the @BaseMechanicalState's
     * \param globalMatrix The product is added into this matrix
     */
    virtual void projectMatrixToGlobalMatrix(const core::MechanicalParams* mparams,
                                             const MappingGraph& mappingGraph,
                                             TMatrix* matrixToProject,
                                             linearalgebra::BaseMatrix* globalMatrix) = 0;

protected:
    explicit BaseMatrixProjectionMethod(const PairMechanicalStates& states);
    BaseMatrixProjectionMethod() = default;
};



#if !defined(SOFA_COMPONENT_LINEARSYSTEM_MATRIXMAPPING_CPP)
extern template class SOFA_COMPONENT_LINEARSYSTEM_API BaseMatrixProjectionMethod<sofa::linearalgebra::CompressedRowSparseMatrix<SReal> >;
#endif

} // namespace sofa::component::linearsystem
