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

#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/component/linearsystem/visitors/AssembleGlobalVectorFromLocalVectorVisitor.h>
#include <sofa/component/linearsystem/visitors/DispatchFromGlobalVectorToLocalVectorVisitor.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>

namespace sofa::component::linearsystem
{

template <class TMatrix, class TVector>
TypedMatrixLinearSystem<TMatrix, TVector>::TypedMatrixLinearSystem()
    : d_matrixChanged(initData(&d_matrixChanged, false, "factorizationInvalidation", "Internal Data indicating a change in the matrix"))
{
    d_matrixChanged.setReadOnly(true);
    d_matrixChanged.setDisplayed(false);
}

template<class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::preAssembleSystem(const core::MechanicalParams* mparams)
{
    allocateSystem();

    {
        SCOPED_TIMER_VARNAME(mappingGraphTimer, "getContributors");

        m_forceFields.clear();
        m_masses.clear();
        m_mechanicalMappings.clear();
        m_projectiveConstraints.clear();

        auto* solveContext = getSolveContext();
        if (solveContext)
        {
            solveContext->getObjects(m_forceFields, core::objectmodel::BaseContext::SearchDirection::SearchDown);
            solveContext->getObjects(m_masses, core::objectmodel::BaseContext::SearchDirection::SearchDown);
            solveContext->getObjects(m_mechanicalMappings, core::objectmodel::BaseContext::SearchDirection::SearchDown);
            solveContext->getObjects(m_projectiveConstraints, core::objectmodel::BaseContext::SearchDirection::SearchDown);
        }

        m_mechanicalMappings.erase(
        std::remove_if(m_mechanicalMappings.begin(), m_mechanicalMappings.end(),
            [](const sofa::core::BaseMapping* mapping) { return !mapping->isMechanical();}),
            m_mechanicalMappings.end());
    }

    {
        SCOPED_TIMER_VARNAME(mappingGraphTimer, "buildMappingGraph");
        // build the mapping graph: this is used to know the relationship between the mechanical states and their associated components
        m_mappingGraph.build(mparams, getSolveContext());
    }

    associateLocalMatrixToComponents(mparams);

    d_matrixChanged.setValue(true);
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::allocateSystem()
{
    m_linearSystem.allocateSystem();
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::resizeVectors(sofa::Size n)
{
    if (m_linearSystem.rhs)
    {
        m_linearSystem.rhs->resize(n);
    }

    if (m_linearSystem.solution)
    {
        m_linearSystem.solution->resize(n);
    }
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::copyLocalVectorToGlobalVector(core::MultiVecDerivId v, TVector* globalVector)
{
    if (globalVector)
    {
        if (sofa::Size(globalVector->size()) < m_mappingGraph.getTotalNbMainDofs())
        {
            globalVector->resize(m_mappingGraph.getTotalNbMainDofs());
        }

        AssembleGlobalVectorFromLocalVectorVisitor(core::execparams::defaultInstance(), m_mappingGraph, v, globalVector)
            .execute(getSolveContext());
    }
}

template <class TMatrix, class TVector>
TMatrix* TypedMatrixLinearSystem<TMatrix, TVector>::getSystemMatrix() const
{
    return m_linearSystem.getMatrix();
}

template <class TMatrix, class TVector>
TVector* TypedMatrixLinearSystem<TMatrix, TVector>::getRHSVector() const
{
    return m_linearSystem.getRHS();
}

template <class TMatrix, class TVector>
TVector* TypedMatrixLinearSystem<TMatrix, TVector>::getSolutionVector() const
{
    return m_linearSystem.getSolution();
}

template <class TMatrix, class TVector>
linearalgebra::BaseMatrix* TypedMatrixLinearSystem<TMatrix, TVector>::getSystemBaseMatrix() const
{
    if constexpr (std::is_base_of_v<sofa::linearalgebra::BaseMatrix, TMatrix>)
    {
        return getSystemMatrix();
    }
    else
    {
        return nullptr;
    }
}
template <class TMatrix, class TVector>
linearalgebra::BaseVector* TypedMatrixLinearSystem<TMatrix, TVector>::getSystemRHSBaseVector() const
{
    if constexpr (std::is_base_of_v<sofa::linearalgebra::BaseVector, TVector>)
    {
        return getRHSVector();
    }
    else
    {
        return nullptr;
    }
}
template <class TMatrix, class TVector>
linearalgebra::BaseVector* TypedMatrixLinearSystem<TMatrix, TVector>::getSystemSolutionBaseVector() const
{
    if constexpr (std::is_base_of_v<sofa::linearalgebra::BaseVector, TVector>)
    {
        return getSolutionVector();
    }
    else
    {
        return nullptr;
    }
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::resizeSystem(sofa::Size n)
{
    m_linearSystem.resizeSystem(n);
    d_matrixChanged.setValue(true);
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::clearSystem()
{
    m_linearSystem.clearSystem();
    d_matrixChanged.setValue(true);
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::setRHS(core::MultiVecDerivId v)
{
    if (!m_mappingGraph.isBuilt()) //note: this check does not make sure the scene graph is different from when the mapping graph has been built
    {
        m_mappingGraph.build(core::execparams::defaultInstance(), getSolveContext());
    }

    copyLocalVectorToGlobalVector(v, getRHSVector());
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::setSystemSolution(core::MultiVecDerivId v)
{
    if (!m_mappingGraph.isBuilt()) //note: this check does not guarantee the scene graph is not different from when the mapping graph has been built
    {
        m_mappingGraph.build(core::execparams::defaultInstance(), getSolveContext());
    }

    if (!v.isNull())
    {
        copyLocalVectorToGlobalVector(v, getSolutionVector());
    }
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::dispatchSystemSolution(core::MultiVecDerivId v)
{
    if (getSolutionVector())
    {
        DispatchFromGlobalVectorToLocalVectorVisitor(core::execparams::defaultInstance(), m_mappingGraph, v, getSolutionVector())
            .execute(getSolveContext());
    }
}

template <class TMatrix, class TVector>
void TypedMatrixLinearSystem<TMatrix, TVector>::dispatchSystemRHS(core::MultiVecDerivId v)
{
    if (getRHSVector())
    {
        DispatchFromGlobalVectorToLocalVectorVisitor(core::execparams::defaultInstance(), m_mappingGraph, v, getRHSVector())
            .execute(getSolveContext());
    }
}

template <class TMatrix, class TVector>
core::objectmodel::BaseContext* TypedMatrixLinearSystem<TMatrix, TVector>::getSolveContext()
{
    auto* linearSolver = this->getContext()->template get<sofa::core::behavior::LinearSolver>(
        core::objectmodel::BaseContext::Local);
    if (linearSolver)
    {
        return linearSolver->getContext();
    }
    linearSolver = this->getContext()->template get<sofa::core::behavior::LinearSolver>(
        core::objectmodel::BaseContext::SearchUp);
    if (linearSolver)
    {
        return linearSolver->getContext();
    }

    return this->getContext();
}

}  // namespace sofa::component::linearsystem
