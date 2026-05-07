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

#include <sofa/component/collision/detection/algorithm/BruteForceBroadPhase.h>
#include <sofa/simulation/task/CpuTask.h>

namespace sofa::core::collision
{
    class Intersection;
    class ElementIntersector;
}

namespace multithreading::component::collision::detection::algorithm
{

class BruteForcePairTest;

/**
 * @brief A parallel implementation of the component BruteForceBroadPhase
 *
 * The work is divided into n tasks executed in parallel. n is the number of threads available in
 * the global thread pool.
 */
class SOFA_MULTITHREADING_PLUGIN_API ParallelBruteForceBroadPhase :
    public sofa::component::collision::detection::algorithm::BruteForceBroadPhase,
    public sofa::simulation::TaskSchedulerUser
{
public:
    SOFA_CLASS(ParallelBruteForceBroadPhase, sofa::component::collision::detection::algorithm::BruteForceBroadPhase);

    void init() override;

    void addCollisionModel(sofa::core::CollisionModel *cm) override;
    void addCollisionModels(const sofa::type::vector<sofa::core::CollisionModel *>& v) override;

protected:
    ParallelBruteForceBroadPhase();
    ~ParallelBruteForceBroadPhase() override = default;

    /// List of tasks executed in parallel.
    /// They are created at each time step, but the memory is not freed
    std::vector<BruteForcePairTest> m_tasks;

public:
    using FirstLastCollisionModelPair = std::pair<FirstLastCollisionModel, FirstLastCollisionModel>;

protected:
    std::vector<FirstLastCollisionModelPair> m_pairs;
};

/**
 * @brief Task meant to be executed in parallel, and performing pair-wise collision tests
 */
class SOFA_MULTITHREADING_PLUGIN_API BruteForcePairTest : public sofa::simulation::CpuTask
{
    using PairIterator = std::vector<ParallelBruteForceBroadPhase::FirstLastCollisionModelPair>::const_iterator;

public:
    BruteForcePairTest(sofa::simulation::CpuTask::Status* status,
                       PairIterator first, PairIterator last,
                       sofa::core::collision::Intersection* intersectionMethod);
    ~BruteForcePairTest() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

    /// After this task is executed, this list contains pairs of collision models which are intersecting
    std::vector<std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> > m_intersectingPairs;

private:

    /// Beginning of a range of pairs of collision models to tests in this task
    PairIterator m_first;
    /// End of a range of pairs of collision models to tests in this task
    PairIterator m_last;

    /// The intersection method used to perform the collision tests
    sofa::core::collision::Intersection* m_intersectionMethod { nullptr };

};

}
