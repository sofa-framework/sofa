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
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/DefaultTaskScheduler.h>

namespace sofa
{

TEST(TaskSchedulerFactory, instantiateNotInFactory)
{
    simulation::TaskSchedulerFactory factory;
    const simulation::TaskScheduler* scheduler = factory.instantiate("notInFactory");
    EXPECT_EQ(scheduler, nullptr);
}



TEST(MainTaskSchedulerFactory, createEmpty)
{
    const simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    EXPECT_NE(dynamic_cast<const simulation::DefaultTaskScheduler*>(scheduler), nullptr);
}

TEST(MainTaskSchedulerFactory, createDefault)
{
    const simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(simulation::DefaultTaskScheduler::name());
    EXPECT_NE(dynamic_cast<const simulation::DefaultTaskScheduler*>(scheduler), nullptr);
}


TEST(MainTaskSchedulerFactory, createNotInFactory)
{
    const simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry("notInFactory");
    EXPECT_EQ(scheduler, nullptr);
}

TEST(MainTaskSchedulerFactory, registerAlreadyInFactory)
{
    simulation::TaskSchedulerFactory factory;
    const bool isRegistered = simulation::MainTaskSchedulerFactory::registerScheduler(
        simulation::DefaultTaskScheduler::name(),
        &simulation::DefaultTaskScheduler::create);
    EXPECT_FALSE(isRegistered);
}

TEST(MainTaskSchedulerFactory, registerNew)
{
    const bool isRegistered = simulation::MainTaskSchedulerFactory::registerScheduler(
        "notTheSameKey", &simulation::DefaultTaskScheduler::create);
    EXPECT_TRUE(isRegistered);
}

}
