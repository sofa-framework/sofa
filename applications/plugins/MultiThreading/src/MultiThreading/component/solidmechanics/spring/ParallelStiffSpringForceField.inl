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

#include <MultiThreading/component/solidmechanics/spring/ParallelStiffSpringForceField.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>

namespace multithreading::component::solidmechanics::spring
{
template <class DataTypes>
void ParallelStiffSpringForceField<DataTypes>::init()
{
    Inherit1::init();
    initTaskScheduler();
}

template <class DataTypes>
void ParallelStiffSpringForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* mparams,
    DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1,
    const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2)
{
    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv> > f1 = sofa::helper::getWriteOnlyAccessor(data_f1);
    sofa::helper::WriteOnlyAccessor<sofa::Data<VecDeriv> > f2 = sofa::helper::getWriteOnlyAccessor(data_f2);

    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();

    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    const sofa::type::vector<Spring>& springs= this->springs.getValue();
    this->dfdx.resize(springs.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;

    std::mutex mutex;

    sofa::simulation::parallelForEachRange(*m_taskScheduler, static_cast<std::size_t>(0), springs.size(),
        [this, &springs, &x1, &v1, &x2, &v2, &mutex, &f1, &f2](const auto& range)
        {
            sofa::type::vector<std::unique_ptr<SpringForce> > springForces;
            springForces.reserve(range.end - range.start);
            for (auto i = range.start; i < range.end; ++i)
            {
                std::unique_ptr<SpringForce> springForce = this->computeSpringForce(x1, v1, x2, v2, springs[i]);
                springForces.push_back(std::move(springForce));
            }

            std::lock_guard lock(mutex);

            std::size_t i = range.start;
            for (auto& springForce : springForces)
            {
                if (springForce)
                {
                    const StiffSpringForce* stiffSpringForce = static_cast<const StiffSpringForce*>(springForce.get());

                    sofa::Index a = springs[i].m1;
                    sofa::Index b = springs[i].m2;

                    DataTypes::setDPos( f1[a], DataTypes::getDPos(f1[a]) + std::get<0>(stiffSpringForce->force)) ;
                    DataTypes::setDPos( f2[b], DataTypes::getDPos(f2[b]) + std::get<1>(stiffSpringForce->force)) ;

                    this->m_potentialEnergy += stiffSpringForce->energy;

                    this->dfdx[i] = stiffSpringForce->dForce_dX;
                }
                else
                {
                    // set derivative to 0
                    this->dfdx[i].clear();
                }
                ++i;
            }
        });
}

template <class DataTypes>
void ParallelStiffSpringForceField<DataTypes>::initTaskScheduler()
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
}
