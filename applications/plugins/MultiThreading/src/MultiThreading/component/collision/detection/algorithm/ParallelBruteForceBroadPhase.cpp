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
#include <MultiThreading/component/collision/detection/algorithm/ParallelBruteForceBroadPhase.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <MultiThreading/ParallelImplementationsRegistry.h>

namespace multithreading::component::collision::detection::algorithm
{

const bool isParallelParallelBruteForceBroadPhaseImplementationRegistered =
    multithreading::ParallelImplementationsRegistry::addEquivalentImplementations("BruteForceBroadPhase", "ParallelBruteForceBroadPhase");

void registerParallelBruteForceBroadPhase(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Parallel version of the collision detection using extensive pair-wise tests performed concurrently.")
                             .add< ParallelBruteForceBroadPhase >());
}

using sofa::helper::ScopedAdvancedTimer;

ParallelBruteForceBroadPhase::ParallelBruteForceBroadPhase()
    : BruteForceBroadPhase()
{}

void ParallelBruteForceBroadPhase::init()
{
    BruteForceBroadPhase::init();

    // initialize the thread pool
    this->initTaskScheduler();
}

void ParallelBruteForceBroadPhase::addCollisionModel(sofa::core::CollisionModel *cm)
{
    if (cm == nullptr || cm->empty())
        return;

    assert(intersectionMethod != nullptr);

    if (boxModel && !intersectWithBoxModel(cm))
    {
        return;
    }

    if (doesSelfCollide(cm))
    {
        // add the collision model to be tested against itself
        cmPairs.emplace_back(cm, cm);
    }

    sofa::core::CollisionModel* finalCollisionModel = cm->getLast();
    for (const auto& model : m_collisionModels)
    {
        m_pairs.emplace_back(FirstLastCollisionModel{cm, finalCollisionModel}, model);
    }

    m_collisionModels.emplace_back(cm, finalCollisionModel);
}

void ParallelBruteForceBroadPhase::addCollisionModels(const sofa::type::vector<sofa::core::CollisionModel *>& v)
{
    SCOPED_TIMER("ParallelBruteForceBroadPhase::addCollisionModels");

    m_pairs.clear();
    BroadPhaseDetection::addCollisionModels(v);

    if (m_pairs.empty())
    {
        return;
    }

    sofa::simulation::CpuTask::Status status;

    {
        SCOPED_TIMER_VARNAME(createTasksTimer, "TasksCreation");

        const auto nbPairs = static_cast<unsigned int>(m_pairs.size());

        const auto nbThreads = std::min(m_taskScheduler->getThreadCount(), nbPairs);
        m_tasks.reserve(nbThreads);

        const auto nbElements = nbPairs / nbThreads;
        auto first = m_pairs.begin();
        auto last = first + nbElements;

        for (unsigned int i = 0; i < nbThreads; ++i)
        {
            if (i == nbThreads - 1)
            {
                last = m_pairs.end();
            }
            m_tasks.emplace_back(&status, first, last, intersectionMethod);
            m_taskScheduler->addTask(&m_tasks.back());

            if (i < nbThreads - 1)
            {
                first += nbElements;
                last += nbElements;
            }
        }
    }

    {
        SCOPED_TIMER_VARNAME(waitTimer, "ParallelTasks");
        m_taskScheduler->workUntilDone(&status);
    }

    // Merge the output of the tasks
    for (const auto& task : m_tasks)
    {
        cmPairs.insert(cmPairs.end(), task.m_intersectingPairs.begin(), task.m_intersectingPairs.end());
    }

    m_tasks.clear();
}

BruteForcePairTest::BruteForcePairTest(sofa::simulation::CpuTask::Status *status,
                                       PairIterator first, PairIterator last,
                                       sofa::core::collision::Intersection* intersectionMethod)
        : sofa::simulation::CpuTask(status)
        , m_intersectingPairs()
        , m_first(first)
        , m_last(last)
        , m_intersectionMethod(intersectionMethod)
{}

sofa::simulation::Task::MemoryAlloc BruteForcePairTest::run()
{
    assert(m_intersectionMethod != nullptr);

    auto it = m_first;
    while(it != m_last)
    {
        auto* cm_1 = it->first.firstCollisionModel;
        auto* lastCm_1 = it->first.lastCollisionModel;
        auto* cm_2 = it->second.firstCollisionModel;
        auto* lastCm_2 = it->second.lastCollisionModel;
        ++it;

        // ignore this pair if both are NOT simulated (inactive)
        if (!cm_1->isSimulated() && !cm_2->isSimulated())
        {
            continue;
        }

        if (!sofa::component::collision::detection::algorithm::BruteForceBroadPhase::keepCollisionBetween(lastCm_1, lastCm_2))
        {
            continue;
        }

        bool swapModels = false;
        sofa::core::collision::ElementIntersector *intersector = m_intersectionMethod->findIntersector(cm_1, cm_2,swapModels);
        if (intersector == nullptr)
        {
            continue;
        }

        if (swapModels)
        {
            std::swap(cm_1, cm_2);
        }

        // Here we assume a single root element is present in both models
        if (intersector->canIntersect(cm_1->begin(), cm_2->begin(), m_intersectionMethod))
        {
            m_intersectingPairs.emplace_back(cm_1, cm_2);
        }
    }

    return sofa::simulation::Task::Stack;
}

}
