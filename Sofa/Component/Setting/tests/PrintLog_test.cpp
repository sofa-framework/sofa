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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/helper/BackTrace.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/component/setting/PrintLog.h>
#include <sofa/simulation/graph/SimpleApi.h>

namespace sofa
{
struct PrintLog_test : public BaseSimulationTest
{
    SceneInstance scene{};

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.Setting");
        sofa::simpleapi::importPlugin("Sofa.Component.SceneUtility");

        simpleapi::createObject(scene.root, "PrintLog", {{"name", "printLog"}});
        simpleapi::createObject(scene.root, "DefaultAnimationLoop", {{"name", "animationLoop"}});
        simpleapi::createObject(scene.root, "DefaultVisualManagerLoop", {{"name", "visualLoop"}, {"printLog", "false"}});

        auto subNode = simpleapi::createChild(scene.root, "subNode");

        simpleapi::createObject(subNode, "InfoComponent", {{"name", "infoComponent"}});

        scene.initScene();
    }
};


TEST_F(PrintLog_test, regularUsage)
{
    auto printLog = scene.root->getObject("printLog");
    EXPECT_NE(printLog, nullptr);

    auto animationLoop = scene.root->getObject("animationLoop");
    EXPECT_NE(animationLoop, nullptr);

    auto visualLoop = scene.root->getObject("visualLoop");
    EXPECT_NE(visualLoop, nullptr);

    auto subNode = scene.root->getChild("subNode");
    EXPECT_NE(subNode, nullptr);

    auto infoComponent = subNode->getObject("infoComponent");
    EXPECT_NE(infoComponent, nullptr);

    EXPECT_TRUE(printLog->f_printLog.getValue());
    EXPECT_TRUE(animationLoop->f_printLog.getValue());
    EXPECT_FALSE(visualLoop->f_printLog.getValue());
    EXPECT_TRUE(infoComponent->f_printLog.getValue());

    EXPECT_EQ(animationLoop->f_printLog.getParent(), &printLog->f_printLog);
    EXPECT_EQ(visualLoop->f_printLog.getParent(), nullptr);
    EXPECT_EQ(infoComponent->f_printLog.getParent(), &printLog->f_printLog);

    // if a new component is added in the graph, the newly component has a link to the PrintLog component
    auto infoComponent2 = simpleapi::createObject(subNode, "InfoComponent", {{"name", "infoComponent2"}});
    EXPECT_EQ(infoComponent2->f_printLog.getParent(), &printLog->f_printLog);
    EXPECT_TRUE(infoComponent2->f_printLog.getValue());


    //if the PrintLog component switch its printLog attribute, it also affects the linked components
    printLog->f_printLog.setValue(false);

    EXPECT_FALSE(printLog->f_printLog.getValue());
    EXPECT_FALSE(animationLoop->f_printLog.getValue());
    EXPECT_FALSE(visualLoop->f_printLog.getValue());
    EXPECT_FALSE(infoComponent->f_printLog.getValue());
    EXPECT_FALSE(infoComponent2->f_printLog.getValue());



    //if the PrintLog component is removed from the graph, the links are broken
    scene.root->removeObject(printLog);

    EXPECT_EQ(animationLoop->f_printLog.getParent(), nullptr);
    EXPECT_EQ(visualLoop->f_printLog.getParent(), nullptr);
    EXPECT_EQ(infoComponent->f_printLog.getParent(), nullptr);
    EXPECT_EQ(infoComponent2->f_printLog.getParent(), nullptr);
}



} // namespace sofa
