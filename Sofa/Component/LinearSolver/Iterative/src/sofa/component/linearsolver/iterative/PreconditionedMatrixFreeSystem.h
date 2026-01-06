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
#include <sofa/component/linearsystem/MatrixFreeSystem.h>
#include <sofa/component/linearsolver/iterative/GraphScatteredTypes.h>

namespace sofa::component::linearsolver::iterative
{

/**
 * A matrix-free linear system that must be used with a preconditioned matrix-free solver
 *
 * This component is like a @MatrixFreeSystem (its base class), but also has a link to another
 * linear system that assembles a matrix. This other linear system is used by a preconditioner
 * in the context of a preconditioned solver.
 */
template <class TMatrix, class TVector>
class PreconditionedMatrixFreeSystem
    : public sofa::component::linearsystem::MatrixFreeSystem<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(PreconditionedMatrixFreeSystem, TMatrix, TVector),
               SOFA_TEMPLATE2(sofa::component::linearsystem::MatrixFreeSystem, TMatrix, TVector));

    void init() override;
    void reset() override;
    void buildSystemMatrix(const core::MechanicalParams* mparams) override;
    void resizeSystem(sofa::Size n) override;
    void clearSystem() override;

    ///< The matrix system of the preconditioner
    SingleLink<MyType, sofa::core::behavior::BaseMatrixLinearSystem, BaseLink::FLAG_DUPLICATE> l_preconditionerSystem;

    Data<unsigned int> d_assemblingRate;

    void reinitAssemblyCounter();

protected:
    PreconditionedMatrixFreeSystem();

    unsigned int m_assemblyCounter {};
};


#if !defined(SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_PRECONDITIONEDMATRIXFREESYSTEM_CPP)
    extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API PreconditionedMatrixFreeSystem<GraphScatteredMatrix, GraphScatteredVector>;
#endif

}
