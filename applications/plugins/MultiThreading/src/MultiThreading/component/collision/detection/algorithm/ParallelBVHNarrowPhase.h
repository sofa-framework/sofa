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

#include <MultiThreading/config.h>
#include <sofa/simulation/task/TaskSchedulerUser.h>

#include <sofa/component/collision/detection/algorithm/BVHNarrowPhase.h>
#include <sofa/simulation/task/CpuTask.h>
#include <unordered_set>

namespace multithreading::component::collision::detection::algorithm
{

class ParallelBVHNarrowPhasePairTask;

class SOFA_MULTITHREADING_PLUGIN_API ParallelBVHNarrowPhase :
    public sofa::component::collision::detection::algorithm::BVHNarrowPhase,
    public sofa::simulation::TaskSchedulerUser
{
public:
    SOFA_CLASS(ParallelBVHNarrowPhase, sofa::component::collision::detection::algorithm::BVHNarrowPhase);

protected:
    ParallelBVHNarrowPhase();

    std::vector<ParallelBVHNarrowPhasePairTask> m_tasks;

    std::unordered_set< sofa::core::topology::BaseMeshTopology* > m_initializedTopology;
    std::set< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > m_initializedPairs;

public:

    void init() override;
    void addCollisionPairs(const sofa::type::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& v) override;

private:

    /// Unlike the sequential algorithm which creates the output on the fly, the parallel implementation
    /// requires to create the outputs before the computation, in order to avoid iterators invalidation
    void createOutput(const sofa::type::vector<std::pair<sofa::core::CollisionModel *, sofa::core::CollisionModel *>> &v);

    /// This function makes sure some topology arrays are initialized. They cannot be initialized concurrently
    void initializeTopology(sofa::core::topology::BaseMeshTopology*);
};

class SOFA_MULTITHREADING_PLUGIN_API ParallelBVHNarrowPhasePairTask : public sofa::simulation::CpuTask
{
public:
    ParallelBVHNarrowPhasePairTask(
            sofa::simulation::CpuTask::Status* status,
            ParallelBVHNarrowPhase* bvhNarrowPhase,
            std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> pair);
    ~ParallelBVHNarrowPhasePairTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

private:

    ParallelBVHNarrowPhase* m_bvhNarrowPhase { nullptr };
    std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> m_pair;
};

} //namespace sofa::component::collision
