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
#include <MultiThreading/component/collision/detection/algorithm/ParallelBVHNarrowPhase.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <MultiThreading/ParallelImplementationsRegistry.h>

namespace multithreading::component::collision::detection::algorithm
{

const bool isParallelBVHNarrowPhaseImplementationRegistered =
    multithreading::ParallelImplementationsRegistry::addEquivalentImplementations("BVHNarrowPhase", "ParallelBVHNarrowPhase");


using sofa::helper::ScopedAdvancedTimer;

int ParallelBVHNarrowPhaseClass = sofa::core::RegisterObject("Narrow phase collision detection based on boundary volume hierarchy")
        .add< ParallelBVHNarrowPhase >()
;

ParallelBVHNarrowPhase::ParallelBVHNarrowPhase()
{}

void ParallelBVHNarrowPhase::init()
{
    NarrowPhaseDetection::init();

    // initialize the thread pool
    initTaskScheduler();
}

void ParallelBVHNarrowPhase::addCollisionPairs(const sofa::type::vector< std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> >& v)
{
    SCOPED_TIMER_VARNAME(addCollisionPairsTimer, "addCollisionPairs");

    if (v.empty())
    {
        return;
    }

    // initialize output
    createOutput(v);

    sofa::simulation::CpuTask::Status status;
    const auto nbPairs = static_cast<unsigned int>(v.size());
    m_tasks.reserve(nbPairs);

    {
        SCOPED_TIMER_VARNAME(createTasksTimer, "TasksCreation");
        for (const auto &pair : v)
        {
            m_tasks.emplace_back(&status, this, pair);
            m_taskScheduler->addTask(&m_tasks.back());
        }
    }

    {
        SCOPED_TIMER_VARNAME(waitTimer, "ParallelTasks");
        m_taskScheduler->workUntilDone(&status);
    }

    m_tasks.clear();

    // m_outputsMap should just be filled in addCollisionPair function
    m_primitiveTestCount = m_outputsMap.size();
}

void ParallelBVHNarrowPhase::createOutput(
        const sofa::type::vector<std::pair<sofa::core::CollisionModel *, sofa::core::CollisionModel *>> &v)
{
    SCOPED_TIMER_VARNAME(createTasksTimer, "OutputCreation");

    for (const auto &pair : v)
    {
        sofa::core::CollisionModel *cm1 = pair.first;
        sofa::core::CollisionModel *cm2 = pair.second;

        sofa::core::CollisionModel *finestCollisionModel1 = cm1->getLast();//get the finest CollisionModel which is not a CubeModel
        sofa::core::CollisionModel *finestCollisionModel2 = cm2->getLast();

        initializeTopology(finestCollisionModel1->getCollisionTopology());
        initializeTopology(finestCollisionModel2->getCollisionTopology());

        bool swapModels = false;
        sofa::core::collision::ElementIntersector *finestIntersector = intersectionMethod->findIntersector(
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
    }
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
        std::pair<sofa::core::CollisionModel*, sofa::core::CollisionModel*> pair)
    : sofa::simulation::CpuTask(status)
    , m_bvhNarrowPhase(bvhNarrowPhase)
    , m_pair(pair)
{}

sofa::simulation::Task::MemoryAlloc ParallelBVHNarrowPhasePairTask::run()
{
    assert(m_bvhNarrowPhase != nullptr);

    m_bvhNarrowPhase->addCollisionPair(m_pair);

    return sofa::simulation::Task::Stack;
}

}
