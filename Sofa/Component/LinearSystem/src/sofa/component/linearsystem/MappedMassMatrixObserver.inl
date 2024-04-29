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
#include <sofa/core/BaseMapping.h>


namespace sofa::component::linearsystem
{

template <class Real>
void MappedMassMatrixObserver<Real>::observe(core::behavior::BaseMass* mass)
{
    m_newObservables |= (mass != m_observedMass);
    m_observedMass = mass;
    m_dataTracker.trackData(m_observedMass->d_componentState);
}

template <class Real>
void MappedMassMatrixObserver<Real>::observe(core::BaseMapping* mapping)
{
    m_dataTracker.trackData(mapping->d_componentState);
}

template <class Real>
void MappedMassMatrixObserver<Real>::observe(core::behavior::BaseMechanicalState* mstate)
{
    m_newObservables |= (mstate != m_mstate);
    m_mstate = mstate;
    m_dataTracker.trackData(m_mstate->d_componentState);
}

template <class Real>
core::behavior::BaseMass* MappedMassMatrixObserver<Real>::getObservableMass() const
{
    return m_observedMass;
}

template <class Real>
core::behavior::BaseMechanicalState* MappedMassMatrixObserver<Real>::getObservableState() const
{
    return m_mstate;
}

template <class Real>
bool MappedMassMatrixObserver<Real>::hasObservableChanged()
{
    const bool hasChanged = m_dataTracker.hasChanged();
    const bool newObservables = m_newObservables;
    m_dataTracker.clean();
    m_newObservables = false;
    return newObservables || hasChanged;
}

}
