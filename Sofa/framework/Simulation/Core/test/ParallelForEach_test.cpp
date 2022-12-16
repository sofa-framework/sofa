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
#include <gtest/gtest.h>
#include <sofa/simulation/DefaultTaskScheduler.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>
#include <sofa/testing/TestMessageHandler.h>

#include <numeric>


namespace sofa
{

std::vector<int> makeTestData(std::size_t nbIntegers = 1024)
{
    std::vector<int> integers(nbIntegers);
    std::iota(integers.begin(), integers.end(), 0);

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i);
    }

    return integers;
}

TEST(ParallelForEach, makeRangesForLoop)
{
    std::vector<int> integers = makeTestData();

    auto ranges = simulation::makeRangesForLoop(integers.begin(), integers.end(), 8u);
    EXPECT_EQ(ranges.size(), 8);

    for (const auto& r : ranges)
    {
        EXPECT_EQ(std::distance(r.start, r.end), integers.size() / 8)
            << "start: " << std::distance(integers.begin(), r.start)
            << ", end: " << std::distance(integers.begin(), r.end);
    }


    ranges = simulation::makeRangesForLoop(integers.begin(), integers.end(), 7u);
    EXPECT_EQ(ranges.size(), 7);

    for (unsigned int i = 0; i < ranges.size() - 1; ++i)
    {
        EXPECT_EQ(std::distance(ranges[i].start, ranges[i].end), integers.size() / 7);
    }
    EXPECT_EQ(std::distance(ranges.back().start, ranges.back().end), integers.size() - 6 * static_cast<std::size_t>(integers.size() / 7));


    ranges = simulation::makeRangesForLoop(integers.begin(), integers.end(), 2048u);
    EXPECT_EQ(ranges.size(), integers.size());

    for (const auto& r : ranges)
    {
        EXPECT_EQ(std::distance(r.start, r.end), 1);
    }
}

TEST(ParallelForEach, incrementVectorLambda)
{
    std::vector<int> integers = makeTestData();

    simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    scheduler->init(0);

    simulation::parallelForEach(*scheduler, integers.begin(), integers.end(), [](int& i) { ++i; });

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
}

TEST(ParallelForEach, incrementVectorFunctor)
{
    struct Functor
    {
        void operator()(int& n)
        {
            ++n;
        }
    };

    std::vector<int> integers = makeTestData();

    simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    scheduler->init(0);

    simulation::parallelForEach(*scheduler, integers.begin(), integers.end(), Functor{});

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
}

TEST(ParallelForEach, nbElementsLessThanThreads)
{
    std::vector<int> integers = makeTestData(3);

    simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    scheduler->init(4);

    simulation::parallelForEach(*scheduler, integers.begin(), integers.end(), [](int& i) { ++i; });

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
}

TEST(ParallelForEach, emptyContainer) //just making sure it does not crash
{
    std::vector<int> integers;

    simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    scheduler->init(0);

    simulation::parallelForEach(*scheduler, integers.begin(), integers.end(), [](int& i) { ++i; });

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
}

TEST(ParallelForEach, integers)
{
    std::vector<int> integers = makeTestData();
    std::vector<int> integers2 = makeTestData();

    simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    scheduler->init(0);

    simulation::parallelForEach(*scheduler, static_cast<std::size_t>(0), integers.size(),
        [&integers, &integers2](const std::size_t& i)
        {
            integers[i]++;
            integers2[i]--;
        });

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers2[i], static_cast<int>(i) - 1);
    }
}

TEST(ParallelForEachRange, nonInitializedTaskScheduler)
{
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    std::vector<int> integers = makeTestData();

    simulation::TaskScheduler* scheduler =
        simulation::MainTaskSchedulerFactory::instantiate(simulation::DefaultTaskScheduler::name());

    EXPECT_MSG_EMIT(Error);
    simulation::parallelForEachRange(*scheduler, integers.begin(), integers.end(),
        [](const auto& range)
        {
            for (auto it = range.start; it != range.end; ++it)
            {
                int& i = *it;
                ++i;
            }
        });

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
}

TEST(ParallelForEachRange, incrementVectorLambda)
{
    std::vector<int> integers = makeTestData();

    simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    scheduler->init(0);

    simulation::parallelForEachRange(*scheduler, integers.begin(), integers.end(),
        [](const auto& range)
        {
            for (auto it = range.start; it != range.end; ++it)
            {
                int& i = *it;
                ++i;
            }
        });

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
}

TEST(ParallelForEachRange, integers)
{
    std::vector<int> integers = makeTestData();
    std::vector<int> integers2 = makeTestData();

    simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    scheduler->init(0);

    simulation::parallelForEachRange(*scheduler, static_cast<std::size_t>(0), integers.size(),
        [&integers, &integers2](const auto& range)
        {
            for (auto it = range.start; it != range.end; ++it)
            {
                ++integers[it];
                --integers2[it];
            }
        });

    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers[i], i + 1);
    }
    for (std::size_t i = 0; i < integers.size(); ++i)
    {
        EXPECT_EQ(integers2[i], static_cast<int>(i) - 1);
    }
}

}
