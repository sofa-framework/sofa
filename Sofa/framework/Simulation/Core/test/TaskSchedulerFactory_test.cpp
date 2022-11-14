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
#include <sofa/simulation/TaskSchedulerFactory.h>
#include <sofa/simulation/DefaultTaskScheduler.h>

namespace sofa
{

TEST(TaskSchedulerFactory, registerAlreadyInFactory)
{
    const bool isRegistered = simulation::TaskSchedulerFactory::registerScheduler(
        simulation::DefaultTaskScheduler::name(),
        &simulation::DefaultTaskScheduler::create);
    EXPECT_FALSE(isRegistered);
}

TEST(TaskSchedulerFactory, createEmpty)
{
    const simulation::TaskScheduler* scheduler = simulation::TaskSchedulerFactory::create();
    EXPECT_NE(dynamic_cast<const simulation::DefaultTaskScheduler*>(scheduler), nullptr);
}

TEST(TaskSchedulerFactory, createDefault)
{
    const simulation::TaskScheduler* scheduler = simulation::TaskSchedulerFactory::create(simulation::DefaultTaskScheduler::name());
    EXPECT_NE(dynamic_cast<const simulation::DefaultTaskScheduler*>(scheduler), nullptr);
}

TEST(TaskSchedulerFactory, createNotInFactory)
{
    const simulation::TaskScheduler* scheduler = simulation::TaskSchedulerFactory::create("notInFactory");
    EXPECT_EQ(scheduler, nullptr);
}

TEST(TaskSchedulerFactory, registerNew)
{
    const bool isRegistered = simulation::TaskSchedulerFactory::registerScheduler(
        "notTheSameKey", &simulation::DefaultTaskScheduler::create);
    EXPECT_TRUE(isRegistered);
}

}
