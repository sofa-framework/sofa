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

#include <MultiThreading/ParallelTetrahedronFEMForceField.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>

namespace sofa::component::forcefield
{

template<class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::init()
{
    Inherit1::init();
    initTaskScheduler();
}

template<class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::initTaskScheduler()
{
    m_taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(m_taskScheduler != nullptr);
    if (m_taskScheduler->getThreadCount() < 1)
    {
        m_taskScheduler->init(0);
        msg_info() << "Task scheduler initialized on " << m_taskScheduler->getThreadCount() << " threads";
    }
    else
    {
        msg_info() << "Task scheduler already initialized on " << m_taskScheduler->getThreadCount() << " threads";
    }
}

template<class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f,
              const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    Inherit1::addForce(mparams, d_f, d_x, d_v);
}

template <class DataTypes>
template <class Function>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForceGeneric(VecDeriv& df, const VecDeriv& dx,
    Real kFactor, const VecElement& indexedElements, Function f)
{
    std::mutex mutex;
    sofa::simulation::parallelForEachRange(*m_taskScheduler, indexedElements.begin(), indexedElements.end(),
           [&indexedElements, this, kFactor, &dx, &df, &f, &mutex](const auto& range)
           {
               auto elementId = std::distance(indexedElements.begin(), range.start);

               VecDeriv& threadLocal_df = m_threadLocal_df[std::this_thread::get_id()];
               threadLocal_df.clear();
               threadLocal_df.resize(df.size());

               for (auto it = range.start; it != range.end; ++it, ++elementId)
               {
                   Index a = (*it)[0];
                   Index b = (*it)[1];
                   Index c = (*it)[2];
                   Index d = (*it)[3];

                   f( threadLocal_df, dx, elementId, a,b,c,d, kFactor );
               }

               std::lock_guard guard(mutex);

               auto it = df.begin();
               for (const auto& d : threadLocal_df)
               {
                   *it++ += d;
               }
           });
}

template <class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForceSmall(VecDeriv& df, const VecDeriv& dx, const Real kFactor, const VecElement& indexedElements)
{
    addDForceGeneric(df, dx, kFactor, indexedElements,
        [this](VecDeriv& f, const VecDeriv& x, Index i, Index a, Index b, Index c, Index d, SReal fact)
        {
            this->applyStiffnessSmall(f, x, i, a, b, c, d, fact);
        });
}

template <class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForceCorotational(VecDeriv& df, const VecDeriv& dx, const Real kFactor, const VecElement& indexedElements)
{
    addDForceGeneric(df, dx, kFactor, indexedElements,
        [this](VecDeriv& f, const VecDeriv& x, Index i, Index a, Index b, Index c, Index d, SReal fact)
        {
            this->applyStiffnessCorotational(f, x, i, a, b, c, d, fact);
        });
}

template<class DataTypes>
void ParallelTetrahedronFEMForceField<DataTypes>::addDForce (const core::MechanicalParams *mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    auto dfAccessor = sofa::helper::getWriteAccessor(d_df);
    VecDeriv& df = dfAccessor.wref();

    const VecDeriv& dx = d_dx.getValue();
    df.resize(dx.size());

    const Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    const auto& indexedElements = *this->_indexedElements;

    if( this->method == Inherit1::SMALL )
    {
        addDForceSmall(df, dx, kFactor, indexedElements);
    }
    else
    {
        addDForceCorotational(df, dx, kFactor, indexedElements);
    }
}


} //namespace sofa::component::forcefield
