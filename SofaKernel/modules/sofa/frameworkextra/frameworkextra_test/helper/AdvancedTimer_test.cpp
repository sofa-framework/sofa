/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/AdvancedTimer.h>

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;
using sofa::core::ExecParams;

#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;

namespace sofa {

/**
 * Test suite for AdvancedTimer
 * @author Lionel Untereiner
 * @date 2017/08/31
 */

struct AdvancedTimerTest: public BaseSimulationTest
{
protected:
	virtual void onSetUp()
	{
		using namespace sofa::helper;

		AdvancedTimer::setEnabled("validID", true);
	}

	void initScene()
	{
		std::stringstream scene ;
		scene << "<?xml version='1.0'?>"
				 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
				 "</Node>                                                                        \n" ;

		//root = SceneLoaderXML::loadFromMemory ("testscene",
		//									scene.str().c_str(),
		//									scene.str().size()) ;
	}

public:
	Node::SPtr root;
};

TEST_F(AdvancedTimerTest, IsEnabled)
{
	ASSERT_TRUE(sofa::helper::AdvancedTimer::isEnabled("validID"));
}

TEST_F(AdvancedTimerTest, SetOutputType)
{
	using namespace sofa::helper;

	AdvancedTimer::setOutputType("validID", "JSON");
	ASSERT_TRUE(AdvancedTimer::getOutputType("validID") == AdvancedTimer::JSON);

	AdvancedTimer::setOutputType("", "JSON");
	ASSERT_TRUE(AdvancedTimer::getOutputType("") == AdvancedTimer::JSON);

	AdvancedTimer::setOutputType("invalid", "JSON");
	ASSERT_TRUE(AdvancedTimer::getOutputType("invalid") == AdvancedTimer::JSON);

	AdvancedTimer::setOutputType("validID", "LJSON");
	ASSERT_TRUE(AdvancedTimer::getOutputType("validID") == AdvancedTimer::LJSON);

	AdvancedTimer::setOutputType("validID", "STDOUT");
	ASSERT_TRUE(AdvancedTimer::getOutputType("validID") == AdvancedTimer::STDOUT);

	AdvancedTimer::setOutputType("validID", "");
	ASSERT_TRUE(AdvancedTimer::getOutputType("validID") == AdvancedTimer::STDOUT);

	AdvancedTimer::setOutputType("validID", "invalidType");
	ASSERT_TRUE(AdvancedTimer::getOutputType("validID") == AdvancedTimer::STDOUT);
}

TEST_F(AdvancedTimerTest, End)
{
	using namespace sofa::helper;
	initScene();

	ASSERT_TRUE(AdvancedTimer::end("validId", root.get()) == std::string(""));
	ASSERT_TRUE(AdvancedTimer::end("", root.get())  == std::string(""));
	EXPECT_NO_FATAL_FAILURE(AdvancedTimer::end("validId", nullptr));
}


} //namespace sofa
