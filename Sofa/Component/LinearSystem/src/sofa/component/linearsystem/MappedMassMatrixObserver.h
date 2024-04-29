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
    /// The provided mass is observed to track cache invalidation
    void observe(core::behavior::BaseMass* mass);

    void observe(core::BaseMapping* mapping);

    /// The provided state (associated to the mass) is observed to track cache invalidation
    void observe(core::behavior::BaseMechanicalState* mstate);

    /// The mass accumulator associated to the observable mass
    BaseAssemblingMatrixAccumulator<core::matrixaccumulator::Contribution::MASS>* accumulator { nullptr };

    /// Return the observable mass
    core::behavior::BaseMass* getObservableMass() const;

    /// Return the observable state
    core::behavior::BaseMechanicalState* getObservableState() const;

    /// Return true if the tracking of the observables noticed a change since the last call
    [[nodiscard]] bool hasObservableChanged();


    std::shared_ptr<linearalgebra::CompressedRowSparseMatrix<Real> > m_invariantMassMatrix;
    Data<linearalgebra::CompressedRowSparseMatrix<Real>> m_invariantProjectedMassMatrix;

protected:

    core::DataTracker m_dataTracker;

    core::behavior::BaseMass* m_observedMass { nullptr };

    /// The state associated to the observable mass
    core::behavior::BaseMechanicalState* m_mstate { nullptr };

    bool m_newObservables = true;
};

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_MAPPEDMASSMATRIXOBSERVER_CPP)
extern template struct SOFA_COMPONENT_LINEARSYSTEM_API MappedMassMatrixObserver<SReal>;
#endif

}
