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
void ParallelStiffSpringForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* mparams,
    DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1,
    const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2)
{
    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    const sofa::type::vector<Spring>& springs= this->springs.getValue();
    this->dfdx.resize(springs.size());
    f1.resize(x1.size());
    f2.resize(x2.size());
    this->m_potentialEnergy = 0;

    for (sofa::Index i=0; i<springs.size(); i++)
    {
        this->addSpringForce(this->m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);
    }

    sofa::simulation::parallelForEachRange(*m_taskScheduler, static_cast<std::size_t>(0), springs.size(),
        [this, &springs, &x1, &v1, &x2, &v2](const auto& range)
        {
            for (auto i = range.start; i < range.end; ++i)
            {
                this->addSpringForce(this->m_potentialEnergy,f1,x1, v1,f2,x2, v2, i, springs[i]);
            }
        });



    data_f1.endEdit();
    data_f2.endEdit();
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
