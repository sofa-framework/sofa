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
#include <MultiThreading/ParallelBVHNarrowPhase.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

namespace sofa::component::collision
{

using sofa::helper::ScopedAdvancedTimer;

int ParallelBVHNarrowPhaseClass = core::RegisterObject("Narrow phase collision detection based on boundary volume hierarchy")
        .add< ParallelBVHNarrowPhase >()
;

ParallelBVHNarrowPhase::ParallelBVHNarrowPhase()
{}

void ParallelBVHNarrowPhase::init()
{
    NarrowPhaseDetection::init();

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

void ParallelBVHNarrowPhase::addCollisionPairs(const sofa::type::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v)
{
    ScopedAdvancedTimer addCollisionPairsTimer("addCollisionPairs");

    if (v.empty())
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

    // initialize output
    createOutput(v);

    sofa::simulation::CpuTask::Status status;
    const auto nbPairs = static_cast<unsigned int>(v.size());
    m_tasks.reserve(nbPairs);

    {
        ScopedAdvancedTimer createTasksTimer("TasksCreation");
        for (const auto &pair : v)
        {
            m_tasks.emplace_back(&status, this, pair);
            taskScheduler->addTask(&m_tasks.back());
        }
    }

    {
        ScopedAdvancedTimer waitTimer("ParallelTasks");
        taskScheduler->workUntilDone(&status);
    }

    m_tasks.clear();

    // m_outputsMap should just be filled in addCollisionPair function
    m_primitiveTestCount = m_outputsMap.size();
}

void ParallelBVHNarrowPhase::createOutput(
        const type::vector<std::pair<core::CollisionModel *, core::CollisionModel *>> &v)
{
    ScopedAdvancedTimer createTasksTimer("OutputCreation");

    for (const auto &pair : v)
    {
        core::CollisionModel *cm1 = pair.first;
        core::CollisionModel *cm2 = pair.second;

        core::CollisionModel *finestCollisionModel1 = cm1->getLast();//get the finest CollisionModel which is not a CubeModel
        core::CollisionModel *finestCollisionModel2 = cm2->getLast();

        initializeTopology(finestCollisionModel1->getCollisionTopology());
        initializeTopology(finestCollisionModel2->getCollisionTopology());

        bool swapModels = false;
        core::collision::ElementIntersector *finestIntersector = intersectionMethod->findIntersector(
                finestCollisionModel1, finestCollisionModel2,
                swapModels);//find the method for the finest CollisionModels
        if (finestIntersector == nullptr)
            continue;
        if (swapModels)
        {
            std::swap(cm1, cm2);
            std::swap(finestCollisionModel1, finestCollisionModel2);
        }

        //force the creation of all Detection Output before the parallel computation
        getDetectionOutputs(finestCollisionModel1, finestCollisionModel2);
    };
}

void ParallelBVHNarrowPhase::initializeTopology(sofa::core::topology::BaseMeshTopology* topology)
{
    if (!topology)
        return;

    auto insertionIt = m_initializedTopology.insert(topology);
    if (insertionIt.second)
    {
        // The following calls force the creation of some topology arrays before the concurrent computing.
        // Those arrays cannot be created on the fly, in a concurrent environment,
        // due to possible race conditions.
        // Depending on the scene graph, it is possible that those calls are not enough.
        if (topology->getNbPoints())
        {
            topology->getTrianglesAroundVertex(0);
        }
        if (topology->getNbTriangles())
        {
            topology->getEdgesInTriangle(0);
        }
        if (topology->getNbEdges())
        {
            topology->getTrianglesAroundEdge(0);
        }
    }
}

ParallelBVHNarrowPhasePairTask::ParallelBVHNarrowPhasePairTask(
        sofa::simulation::CpuTask::Status* status,
        ParallelBVHNarrowPhase* bvhNarrowPhase,
        std::pair<core::CollisionModel*, core::CollisionModel*> pair)
    : sofa::simulation::CpuTask(status)
    , m_bvhNarrowPhase(bvhNarrowPhase)
    , m_pair(pair)
{}

sofa::simulation::Task::MemoryAlloc ParallelBVHNarrowPhasePairTask::run()
{
    assert(m_bvhNarrowPhase != nullptr);

    m_bvhNarrowPhase->addCollisionPair(m_pair);

    return simulation::Task::Stack;
}

}
