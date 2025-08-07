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

#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>

namespace sofa::component::linearsystem
{

/**
 * Component acting like a linear system, but delegating the linear system functionalities to a list of real linear systems.
 *
 * Using this component allows assembling more than one global matrix. It is useful when only a partial assembly is
 * necessary. For example, the first linear system is the global one, and the second one could be only the stiffness
 * matrix.
 */
template<class TMatrix, class TVector>
class CompositeLinearSystem : public TypedMatrixLinearSystem<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CompositeLinearSystem, TMatrix, TVector), SOFA_TEMPLATE2(TypedMatrixLinearSystem, TMatrix, TVector));

protected:
    CompositeLinearSystem();

public:
    void init() override;

    TMatrix* getSystemMatrix() const override;
    TVector* getRHSVector() const override;
    TVector* getSolutionVector() const override;
    [[nodiscard]] sofa::linearalgebra::BaseMatrix* getSystemBaseMatrix() const override;
    void buildSystemMatrix(const core::MechanicalParams* mparams) override;
    void resizeSystem(sofa::Size n) override;
    void clearSystem() override;
    void setRHS(core::MultiVecDerivId v) override;
    void setSystemSolution(core::MultiVecDerivId v) override;
    void dispatchSystemSolution(core::MultiVecDerivId v) override;
    void dispatchSystemRHS(core::MultiVecDerivId v) override;

protected:
    ///< List of linear systems to assemble
    MultiLink < MyType, sofa::core::behavior::BaseMatrixLinearSystem, BaseLink::FLAG_DUPLICATE > l_linearSystems;

    ///< Among the list of linear systems, which one is to be used by the linear solver
    SingleLink < MyType, TypedMatrixLinearSystem<TMatrix, TVector>, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK > l_solverLinearSystem;
};

}
