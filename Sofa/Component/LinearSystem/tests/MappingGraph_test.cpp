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
#include <sofa/testing/BaseTest.h>
#include <sofa/component/linearsystem/MappingGraph.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/mapping/linear/SubsetMapping.h>
#include <sofa/simulation/graph/DAGSimulation.h>

TEST(MappingGraph, noBuild)
{
    const sofa::component::linearsystem::MappingGraph graph;

    EXPECT_FALSE(graph.isBuilt());
    EXPECT_EQ(graph.getRootNode(), nullptr);
    EXPECT_TRUE(graph.getMainMechanicalStates().empty());

    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseMechanicalState*)nullptr).empty());
    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseForceField*)nullptr).empty());
    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseMass*)nullptr).empty());

    EXPECT_TRUE(graph.makeComponentGroups(sofa::core::MechanicalParams::defaultInstance()).empty());
    EXPECT_FALSE(graph.hasAnyMapping());
    EXPECT_EQ(graph.getTotalNbMainDofs(), 0);
}

TEST(MappingGraph, nullRootNode)
{
    sofa::component::linearsystem::MappingGraph graph;
    graph.build(sofa::core::MechanicalParams::defaultInstance(), nullptr);

    EXPECT_FALSE(graph.isBuilt());
    EXPECT_EQ(graph.getRootNode(), nullptr);
    EXPECT_TRUE(graph.getMainMechanicalStates().empty());

    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseMechanicalState*)nullptr).empty());
    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseForceField*)nullptr).empty());
    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseMass*)nullptr).empty());

    EXPECT_TRUE(graph.makeComponentGroups(sofa::core::MechanicalParams::defaultInstance()).empty());
    EXPECT_FALSE(graph.hasAnyMapping());
    EXPECT_EQ(graph.getTotalNbMainDofs(), 0);
}

TEST(MappingGraph, emptyRootNode)
{
    const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();

    sofa::component::linearsystem::MappingGraph graph;
    graph.build(sofa::core::MechanicalParams::defaultInstance(), root.get());

    EXPECT_TRUE(graph.isBuilt());
    EXPECT_EQ(graph.getRootNode(), root.get());
    EXPECT_TRUE(graph.getMainMechanicalStates().empty());

    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseMechanicalState*)nullptr).empty());
    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseForceField*)nullptr).empty());
    EXPECT_TRUE(graph.getTopMostMechanicalStates((sofa::core::behavior::BaseMass*)nullptr).empty());

    EXPECT_TRUE(graph.makeComponentGroups(sofa::core::MechanicalParams::defaultInstance()).empty());
    EXPECT_FALSE(graph.hasAnyMapping());
    EXPECT_EQ(graph.getTotalNbMainDofs(), 0);
}

TEST(MappingGraph, oneMechanicalObject)
{
    const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();

    const auto mstate = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate);
    mstate->resize(10);

    sofa::component::linearsystem::MappingGraph graph;
    graph.build(sofa::core::MechanicalParams::defaultInstance(), root.get());

    EXPECT_TRUE(graph.isBuilt());
    EXPECT_EQ(graph.getRootNode(), root.get());
    EXPECT_FALSE(graph.getMainMechanicalStates().empty());

    EXPECT_EQ(graph.getTopMostMechanicalStates(mstate.get()), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{mstate.get()});

    EXPECT_TRUE(graph.makeComponentGroups(sofa::core::MechanicalParams::defaultInstance()).empty());
    EXPECT_FALSE(graph.hasAnyMapping());
    EXPECT_EQ(graph.getTotalNbMainDofs(), 30);
}

TEST(MappingGraph, twoMechanicalObject)
{
    const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();

    const auto mstate1 = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate1);
    mstate1->resize(10);

    const auto mstate2 = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate2);
    mstate2->resize(2);

    sofa::component::linearsystem::MappingGraph graph;
    graph.build(sofa::core::MechanicalParams::defaultInstance(), root.get());

    EXPECT_TRUE(graph.isBuilt());
    EXPECT_EQ(graph.getRootNode(), root.get());
    const sofa::type::vector<sofa::core::behavior::BaseMechanicalState*> allMStates {mstate1.get(), mstate2.get()};
    EXPECT_EQ(graph.getMainMechanicalStates(), allMStates);

    EXPECT_EQ(graph.getTopMostMechanicalStates(mstate1.get()), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{mstate1.get()});
    EXPECT_EQ(graph.getTopMostMechanicalStates(mstate2.get()), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{mstate2.get()});

    EXPECT_TRUE(graph.makeComponentGroups(sofa::core::MechanicalParams::defaultInstance()).empty());
    EXPECT_FALSE(graph.hasAnyMapping());
    EXPECT_EQ(graph.getTotalNbMainDofs(), 36);
}

TEST(MappingGraph, oneMapping)
{
    const sofa::simulation::Node::SPtr root = sofa::core::objectmodel::New<sofa::simulation::Node>();

    const auto mstate1 = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate1);
    mstate1->resize(10);

    const auto mapping = sofa::core::objectmodel::New<sofa::component::mapping::linear::SubsetMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types> >();
    root->addObject(mapping);

    const auto mstate2 = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(mstate2);
    mstate2->resize(2);

    mapping->setFrom(mstate1.get());
    mapping->setTo(mstate2.get());

    sofa::component::linearsystem::MappingGraph graph;
    graph.build(sofa::core::MechanicalParams::defaultInstance(), root.get());

    EXPECT_TRUE(graph.isBuilt());
    EXPECT_EQ(graph.getRootNode(), root.get());
    EXPECT_EQ(graph.getMainMechanicalStates(), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{mstate1.get()});

    EXPECT_EQ(graph.getTopMostMechanicalStates(mstate1.get()), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{mstate1.get()});
    EXPECT_EQ(graph.getTopMostMechanicalStates(mstate2.get()), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{mstate1.get()});

    EXPECT_TRUE(graph.makeComponentGroups(sofa::core::MechanicalParams::defaultInstance()).empty());
    EXPECT_TRUE(graph.hasAnyMapping());
    EXPECT_EQ(graph.getTotalNbMainDofs(), 30);
}

TEST(MappingGraph, diamondMapping)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    sofa::simulation::Simulation* simulation = sofa::simulation::getSimulation();

    sofa::simulation::Node::SPtr root = simulation->createNewGraph("root");
    EXPECT_EQ(root->getName(), "root");

    const auto plugins = sofa::testing::makeScopedPlugin({Sofa.Component.Mapping.Linear});

    const auto top = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    root->addObject(top);
    top->setName("top");
    top->resize(3);

    sofa::simulation::Node::SPtr leftNode = sofa::simpleapi::createChild(root, "left");
    sofa::simulation::Node::SPtr rightNode = sofa::simpleapi::createChild(root, "right");

    const auto left = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    leftNode->addObject(left);
    left->setName("left");

    const auto right = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    rightNode->addObject(right);
    right->setName("right");

    sofa::simpleapi::createObject(rightNode, "SubsetMapping", {
        {"name", "mapping"}, {"indices","0"}, {"input", "@../top"}, {"output", "@right"}
    });
    sofa::simpleapi::createObject(leftNode, "SubsetMapping", {
        {"name", "mapping"}, {"indices","2"}, {"input", "@../top"}, {"output", "@left"}
    });

    sofa::simulation::Node::SPtr bottomNode = sofa::simpleapi::createChild(leftNode, "bottom");
    rightNode->addChild(bottomNode);

    const auto bottom = sofa::core::objectmodel::New<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> >();
    bottomNode->addObject(bottom);
    bottom->setName("bottom");

    sofa::simpleapi::createObject(bottomNode, "SubsetMultiMapping", {
        {"name", "mapping"}, {"indices","0 0 1 0"}, {"template", "Vec3d,Vec3d"},
        {"input", "@/left/left @/right/right"}, {"output", "@bottom"}
    });

    sofa::component::linearsystem::MappingGraph graph;
    graph.build(sofa::core::MechanicalParams::defaultInstance(), root.get());

    EXPECT_TRUE(graph.isBuilt());
    EXPECT_EQ(graph.getRootNode(), root.get());
    EXPECT_EQ(graph.getMainMechanicalStates(), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{top.get()});

    EXPECT_EQ(graph.getTopMostMechanicalStates(left.get()), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{top.get()});
    EXPECT_EQ(graph.getTopMostMechanicalStates(right.get()), sofa::type::vector<sofa::core::behavior::BaseMechanicalState*>{top.get()});
    const sofa::type::vector<sofa::core::behavior::BaseMechanicalState*> expectedList {top.get(), top.get()};
    EXPECT_EQ(graph.getTopMostMechanicalStates(bottom.get()), expectedList);

    EXPECT_TRUE(graph.makeComponentGroups(sofa::core::MechanicalParams::defaultInstance()).empty());
    EXPECT_TRUE(graph.hasAnyMapping());
    EXPECT_EQ(graph.getTotalNbMainDofs(), 3 * 3);

    EXPECT_EQ(graph.getPositionInGlobalMatrix(top.get()), sofa::type::Vec2u{});

    {
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(graph.getPositionInGlobalMatrix(left.get()), sofa::type::Vec2u{});
    }

    {
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(graph.getPositionInGlobalMatrix(right.get()), sofa::type::Vec2u{});
    }

    {
        EXPECT_MSG_EMIT(Error);
        EXPECT_EQ(graph.getPositionInGlobalMatrix(bottom.get()), sofa::type::Vec2u{});
    }

}
