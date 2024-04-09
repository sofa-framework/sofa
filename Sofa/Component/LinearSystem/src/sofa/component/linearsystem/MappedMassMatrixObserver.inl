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
#include <sofa/component/linearsystem/MappedMassMatrixObserver.h>

namespace sofa::component::linearsystem
{

template <class Real>
MappedMassMatrixObserver<Real>::MappedMassMatrixObserver()
{
    dataTracker.addOutput(&this->m_invariantProjectedMassMatrix);
}

template <class Real>
void MappedMassMatrixObserver<Real>::observe(core::behavior::BaseMass* mass)
{
    m_observedMass = mass;
    this->trackMatrixChangesFrom(&m_observedMass->d_recomputeCachedMassMatrix);
}

template <class Real>
core::behavior::BaseMass* MappedMassMatrixObserver<Real>::getObservableMass() const
{
    return m_observedMass;
}

template<class Real>
void MappedMassMatrixObserver<Real>::trackMatrixChangesFrom(core::objectmodel::DDGNode* input)
{
    dataTracker.addInput(input);
}

template <class Real>
void MappedMassMatrixObserver<Real>::setRecomputionMappedMassMatrix(
    std::function<sofa::core::objectmodel::ComponentState(const core::DataTracker&)> f)
{
    dataTracker.setCallback(f);
}

}
