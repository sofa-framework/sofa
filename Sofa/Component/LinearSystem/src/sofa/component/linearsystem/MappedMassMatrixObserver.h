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
#include <sofa/core/CachedDataObserver.h>
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
struct MappedMassMatrixObserver : core::CachedDataObserver
{
    core::behavior::BaseMass* observedMass { nullptr };
    BaseAssemblingMatrixAccumulator<core::matrixaccumulator::Contribution::MASS>* accumulator { nullptr };

    void postObservableDestroyed(core::CachedDataObservable* observable) override;

    std::shared_ptr<linearalgebra::CompressedRowSparseMatrix<Real> > m_invariantMassMatrix =
        std::make_shared<linearalgebra::CompressedRowSparseMatrix<Real> >();

    std::shared_ptr<linearalgebra::CompressedRowSparseMatrix<Real> > m_invariantProjectedMassMatrix =
        std::make_shared<linearalgebra::CompressedRowSparseMatrix<Real> >();

    core::behavior::BaseMechanicalState* mstate { nullptr };
};

#if !defined(SOFA_COMPONENT_LINEARSYSTEM_MAPPEDMASSMATRIXOBSERVER_CPP)
extern template struct SOFA_COMPONENT_LINEARSYSTEM_API MappedMassMatrixObserver<SReal>;
#endif

}
