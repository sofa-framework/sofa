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

#include <SofaBaseCollision/BruteForceBroadPhase.h>
#include <sofa/simulation/Task.h>

namespace sofa::core::collision
{
    class Intersection;
    class ElementIntersector;
}

namespace sofa::component::collision
{

class BruteForcePairTest;

class SOFA_MULTITHREADING_PLUGIN_API ParallelBruteForceBroadPhase : public BruteForceBroadPhase
{
public:
    SOFA_CLASS(ParallelBruteForceBroadPhase, BruteForceBroadPhase);

    void init() override;

    void addCollisionModel(core::CollisionModel *cm) override;
    void addCollisionModels(const sofa::helper::vector<core::CollisionModel *>& v) override;

protected:
    ParallelBruteForceBroadPhase();
    ~ParallelBruteForceBroadPhase() override = default;

    std::vector<BruteForcePairTest> m_tasks;
    sofa::simulation::CpuTask::Status m_status;

public:
    using Pair = std::pair<FirstLastCollisionModel, FirstLastCollisionModel>;

protected:
    std::vector<Pair> m_pairs;
};

class SOFA_SOFABASECOLLISION_API BruteForcePairTest : public sofa::simulation::CpuTask
{
    using PairIterator = std::vector<ParallelBruteForceBroadPhase::Pair>::const_iterator;

public:
    BruteForcePairTest(sofa::simulation::CpuTask::Status* status,
                       PairIterator first, PairIterator last,
                       core::collision::Intersection* intersectionMethod);
    ~BruteForcePairTest() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

    std::vector<std::pair<core::CollisionModel*, core::CollisionModel*> > m_intersectingPairs;

private:

    PairIterator m_first;
    PairIterator m_last;

    core::collision::Intersection* m_intersectionMethod { nullptr };

};

}