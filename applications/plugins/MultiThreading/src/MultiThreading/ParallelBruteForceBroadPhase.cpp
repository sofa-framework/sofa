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
#include <MultiThreading/ParallelBruteForceBroadPhase.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

namespace sofa::component::collision
{

using sofa::helper::ScopedAdvancedTimer;

int ParallelBruteForceBroadPhaseClass = core::RegisterObject("Collision detection using extensive pair-wise tests performed in parallel")
        .add< ParallelBruteForceBroadPhase >()
;

ParallelBruteForceBroadPhase::ParallelBruteForceBroadPhase()
    : BruteForceBroadPhase()
{}

void ParallelBruteForceBroadPhase::init()
{
    BruteForceBroadPhase::init();

    // initialize the thread pool

    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);
    if (taskScheduler->getThreadCount() < 1)
    {
        taskScheduler->init(0);
        msg_info() << "Task scheduler initialized on " << taskScheduler->getThreadCount() << " threads";
    }
    else
    {
        msg_info() << "Task scheduler already initialized on " << taskScheduler->getThreadCount() << " threads";
    }
}

void ParallelBruteForceBroadPhase::addCollisionModel(core::CollisionModel *cm)
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

    core::CollisionModel* finalCollisionModel = cm->getLast();
    for (const auto& model : m_collisionModels)
    {
        m_pairs.emplace_back(FirstLastCollisionModel{cm, finalCollisionModel}, model);
    }

    m_collisionModels.emplace_back(cm, finalCollisionModel);
}

void ParallelBruteForceBroadPhase::addCollisionModels(const sofa::type::vector<core::CollisionModel *>& v)
{
    ScopedAdvancedTimer timer("ParallelBruteForceBroadPhase::addCollisionModels");

    m_pairs.clear();
    BroadPhaseDetection::addCollisionModels(v);

    if (m_pairs.empty())
    {
        return;
    }

    auto *taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);

    if (taskScheduler->getThreadCount() == 0)
    {
        msg_error() << "Task scheduler not correctly initialized";
        return;
    }

    sofa::simulation::CpuTask::Status status;

    {
        ScopedAdvancedTimer createTasksTimer("TasksCreation");

        const auto nbPairs = static_cast<unsigned int>(m_pairs.size());

        const auto nbThreads = std::min(taskScheduler->getThreadCount(), nbPairs);
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
            taskScheduler->addTask(&m_tasks.back());
            first += nbElements;
            last += nbElements;
        }
    }

    {
        ScopedAdvancedTimer waitTimer("ParallelTasks");
        taskScheduler->workUntilDone(&status);
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
                                       core::collision::Intersection* intersectionMethod)
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

        if (!detection::algorithm::BruteForceBroadPhase::keepCollisionBetween(lastCm_1, lastCm_2))
        {
            continue;
        }

        bool swapModels = false;
        core::collision::ElementIntersector *intersector = m_intersectionMethod->findIntersector(cm_1, cm_2,swapModels);
        if (intersector == nullptr)
        {
            continue;
        }

        if (swapModels)
        {
            std::swap(cm_1, cm_2);
        }

        // Here we assume a single root element is present in both models
        if (intersector->canIntersect(cm_1->begin(), cm_2->begin()))
        {
            m_intersectingPairs.emplace_back(cm_1, cm_2);
        }
    }

    return simulation::Task::Stack;
}

}
