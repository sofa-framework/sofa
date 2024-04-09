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
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/component/linearsystem/config.h>
#include <sofa/component/linearsystem/matrixaccumulators/BaseAssemblingMatrixAccumulator.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>


namespace sofa::component::linearsystem
{

/**
 * Pre-compute the mapped mass matrix assuming it is constant and store it.
 * Support cache invalidation
 */
template<class Real>
struct MappedMassMatrixObserver
{
    MappedMassMatrixObserver();;
    MappedMassMatrixObserver(const MappedMassMatrixObserver&) = default;

    void observe(core::behavior::BaseMass* mass);
    core::behavior::BaseMass* getObservableMass() const;

    BaseAssemblingMatrixAccumulator<core::matrixaccumulator::Contribution::MASS>* accumulator { nullptr };

    std::shared_ptr<linearalgebra::CompressedRowSparseMatrix<Real> > m_invariantMassMatrix;
    Data<linearalgebra::CompressedRowSparseMatrix<Real>> m_invariantProjectedMassMatrix;

    core::behavior::BaseMechanicalState* mstate { nullptr };

    void trackMatrixChangesFrom(core::objectmodel::DDGNode* input);
    void setRecomputionMappedMassMatrix(std::function<sofa::core::objectmodel::ComponentState(const core::DataTracker&)> f);

protected:
    core::DataTrackerCallback dataTracker;

    core::behavior::BaseMass* m_observedMass { nullptr };
};

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_MAPPEDMASSMATRIXOBSERVER_CPP)
extern template struct SOFA_COMPONENT_LINEARSYSTEM_API MappedMassMatrixObserver<SReal>;
#endif

}
