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
#include <sofa/Modules.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/MappingGraph.h>

namespace sofa
{

TEST(MappingGraph, DefaultConstructor)
{
    sofa::simulation::MappingGraph mappingGraph;
    EXPECT_FALSE(mappingGraph.isBuilt());
}

TEST(MappingGraph, Build)
{
    sofa::simulation::MappingGraph mappingGraph;
    sofa::simulation::MappingGraph::InputLists input;
    mappingGraph.build(input);
    EXPECT_TRUE(mappingGraph.isBuilt());
}

struct CollectNamesVisitor : public sofa::simulation::MappingGraphVisitor
{
    void visit(core::BaseMapping& mapping) override
    {
        names.push_back(mapping.getName());
    }

    void visit(core::behavior::BaseMechanicalState& state) override
    {
        names.push_back(state.getName());
    }

    void visit(core::behavior::BaseForceField& ff) override
    {
        names.push_back(ff.getName());
    }

    void visit(core::behavior::BaseMass& mass) override
    {
        names.push_back(mass.getName());
    }

    std::vector<std::string> names;
};

TEST(MappingGraph, SingleState)
{
    const sofa::simulation::Node::SPtr root = sofa::simpleapi::createRootNode(sofa::simulation::getSimulation(), "root");

    sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"name", "state"}});

    auto inputs = sofa::simulation::MappingGraph::InputLists::makeFromNode(root.get());
    ASSERT_EQ(inputs.mechanicalStates.size(), 1);
    sofa::simulation::MappingGraph mappingGraph(inputs);
    ASSERT_TRUE(mappingGraph.isBuilt());

    CollectNamesVisitor visitor;

    mappingGraph.traverseTopDown(visitor);
    ASSERT_EQ(visitor.names.size(), 1);
    EXPECT_EQ(visitor.names[0], "state");
}

TEST(MappingGraph, SingleMappingInSingleNode)
{
    const sofa::simulation::Node::SPtr root = sofa::simpleapi::createRootNode(sofa::simulation::getSimulation(), "root");

    sofa::simpleapi::importPlugin(Sofa.Component.Mapping.Linear);
    sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);

    sofa::simpleapi::createObject(root, "MechanicalObject", {{"name", "state1"}});
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"name", "state2"}});
    sofa::simpleapi::createObject(root, "IdentityMapping", {{"name", "mapping"}, {"input", "@state1"}, {"output", "@state2"}});

    auto inputs = sofa::simulation::MappingGraph::InputLists::makeFromNode(root.get());
    ASSERT_EQ(inputs.mappings.size(), 1);
    ASSERT_EQ(inputs.mechanicalStates.size(), 2);
    sofa::simulation::MappingGraph mappingGraph(inputs);
    ASSERT_TRUE(mappingGraph.isBuilt());

    CollectNamesVisitor visitor;

    mappingGraph.traverseTopDown(visitor);
    ASSERT_EQ(visitor.names.size(), 3);
    EXPECT_EQ(visitor.names[0], "state1");
    EXPECT_EQ(visitor.names[1], "mapping");
    EXPECT_EQ(visitor.names[2], "state2");

    visitor.names.clear();
    mappingGraph.traverseBottomUp(visitor);
    ASSERT_EQ(visitor.names.size(), 3);
    EXPECT_EQ(visitor.names[0], "state2");
    EXPECT_EQ(visitor.names[1], "mapping");
    EXPECT_EQ(visitor.names[2], "state1");
}

TEST(MappingGraph, SingleMappingWithIntermediateNode)
{
    const sofa::simulation::Node::SPtr root = sofa::simpleapi::createRootNode(sofa::simulation::getSimulation(), "root");

    sofa::simpleapi::importPlugin(Sofa.Component.Mapping.Linear);
    sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);

    sofa::simpleapi::createObject(root, "MechanicalObject", {{"name", "state1"}});
    const auto node1 = root->createChild("node1");
    sofa::simpleapi::createObject(node1, "MechanicalObject", {{"name", "state2"}});
    sofa::simpleapi::createObject(node1, "IdentityMapping", {{"name", "mapping"}, {"input", "@state1"}, {"output", "@state2"}});

    auto inputs = sofa::simulation::MappingGraph::InputLists::makeFromNode(root);
    ASSERT_EQ(inputs.mappings.size(), 1);
    ASSERT_EQ(inputs.mechanicalStates.size(), 2);
    sofa::simulation::MappingGraph mappingGraph(inputs);
    ASSERT_TRUE(mappingGraph.isBuilt());

    CollectNamesVisitor visitor;

    mappingGraph.traverseTopDown(visitor);
    ASSERT_EQ(visitor.names.size(), 3);
    EXPECT_EQ(visitor.names[0], "state1");
    EXPECT_EQ(visitor.names[1], "mapping");
    EXPECT_EQ(visitor.names[2], "state2");

    visitor.names.clear();
    mappingGraph.traverseBottomUp(visitor);
    ASSERT_EQ(visitor.names.size(), 3);
    EXPECT_EQ(visitor.names[0], "state2");
    EXPECT_EQ(visitor.names[1], "mapping");
    EXPECT_EQ(visitor.names[2], "state1");
}

TEST(MappingGraph, SingleMappingWithIntermediateNodeInverseInputOutput)
{
    const sofa::simulation::Node::SPtr root = sofa::simpleapi::createRootNode(sofa::simulation::getSimulation(), "root");

    sofa::simpleapi::importPlugin(Sofa.Component.Mapping.Linear);
    sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);

    sofa::simpleapi::createObject(root, "MechanicalObject", {{"name", "state1"}});
    const auto node1 = root->createChild("node1");
    sofa::simpleapi::createObject(node1, "MechanicalObject", {{"name", "state2"}});
    sofa::simpleapi::createObject(node1, "IdentityMapping", {{"name", "mapping"}, {"input", "@state2"}, {"output", "@state1"}});

    auto inputs = sofa::simulation::MappingGraph::InputLists::makeFromNode(root);
    ASSERT_EQ(inputs.mappings.size(), 1);
    ASSERT_EQ(inputs.mechanicalStates.size(), 2);
    sofa::simulation::MappingGraph mappingGraph(inputs);
    ASSERT_TRUE(mappingGraph.isBuilt());

    CollectNamesVisitor visitor;

    mappingGraph.traverseTopDown(visitor);
    ASSERT_EQ(visitor.names.size(), 3);
    EXPECT_EQ(visitor.names[0], "state2");
    EXPECT_EQ(visitor.names[1], "mapping");
    EXPECT_EQ(visitor.names[2], "state1");

    visitor.names.clear();
    mappingGraph.traverseBottomUp(visitor);
    ASSERT_EQ(visitor.names.size(), 3);
    EXPECT_EQ(visitor.names[0], "state1");
    EXPECT_EQ(visitor.names[1], "mapping");
    EXPECT_EQ(visitor.names[2], "state2");
}

/**
 * @brief Sets up the complex graph environment for testing, creating nodes and components.
 * 
 * @return std::tuple<sofa::simulation::Node::SPtr, sofa::simulation::MappingGraph::InputLists> A tuple containing the root node pointer and collected input lists.
 */
auto setupComplexGraphEnvironment() -> std::pair<const sofa::simulation::Node::SPtr, sofa::simulation::MappingGraph::InputLists>
{
    // Setup common plugins required for both complex graph tests
    sofa::simpleapi::importPlugin(Sofa.Component.Mapping.Linear);
    sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);
    sofa::simpleapi::importPlugin(Sofa.Component.MechanicalLoad);
    sofa::simpleapi::importPlugin(Sofa.Component.Mass);

    const sofa::simulation::Node::SPtr root = sofa::simpleapi::createRootNode(sofa::simulation::getSimulation(), "root");

    // Components on the root node (state1)
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"name", "state1"}});
    sofa::simpleapi::createObject(root, "ConstantForceField", {{"name", "ff1"}, {"state", "@state1"}, {"forces", "1 0 0"}});
    sofa::simpleapi::createObject(root, "UniformMass", {{"name", "mass1"}});
    
    // Components on the child node (state2)
    const auto node1 = root->createChild("node1");
    sofa::simpleapi::createObject(node1, "MechanicalObject", {{"name", "state2"}});
    sofa::simpleapi::createObject(node1, "ConstantForceField", {{"name", "ff2"}, {"state", "@state2"}, {"forces", "1 0 0"}});
    sofa::simpleapi::createObject(node1, "UniformMass", {{"name", "mass2"}});

    // Mapping connecting state1 to state2
    sofa::simpleapi::createObject(node1, "IdentityMapping", {{"name", "mapping"}, {"input", "@state1"}, {"output", "@state2"}});

    // Initialize the root node structure
    sofa::simulation::node::initRoot(root.get());

    auto inputs = sofa::simulation::MappingGraph::InputLists::makeFromNode(root);
    return {root, inputs};
}


TEST(MappingGraph, ComplexGraph)
{
    // Setup environment using helper function
    auto [root, inputs] = setupComplexGraphEnvironment();

    ASSERT_EQ(inputs.mappings.size(), 1);
    ASSERT_EQ(inputs.mechanicalStates.size(), 2);
    ASSERT_EQ(inputs.forceFields.size(), 4);
    sofa::simulation::MappingGraph mappingGraph(inputs);
    ASSERT_TRUE(mappingGraph.isBuilt());

    CollectNamesVisitor visitor;

    // Top Down Traversal Check
    mappingGraph.traverseTopDown(visitor);
    ASSERT_EQ(visitor.names.size(), 9); // 9 and not 7 because a UniformMass is a BaseMass and also a BaseForceField

    EXPECT_EQ(visitor.names[0], "state1");
    EXPECT_EQ(visitor.names[1], "ff1");
    EXPECT_EQ(visitor.names[2], "mass1");
    EXPECT_EQ(visitor.names[3], "mass1");

    EXPECT_EQ(visitor.names[4], "mapping");
    EXPECT_EQ(visitor.names[5], "state2");
    EXPECT_EQ(visitor.names[6], "ff2");
    EXPECT_EQ(visitor.names[7], "mass2");
    EXPECT_EQ(visitor.names[8], "mass2");

    visitor.names.clear();
    // Bottom Up Traversal Check
    mappingGraph.traverseBottomUp(visitor);
    ASSERT_EQ(visitor.names.size(), 9);

    EXPECT_EQ(visitor.names[0], "mass2");
    EXPECT_EQ(visitor.names[1], "mass2");
    EXPECT_EQ(visitor.names[2], "ff2");
    EXPECT_EQ(visitor.names[3], "state2");

    EXPECT_EQ(visitor.names[4], "mapping");
    EXPECT_EQ(visitor.names[5], "mass1");
    EXPECT_EQ(visitor.names[6], "mass1");
    EXPECT_EQ(visitor.names[7], "ff1");
    EXPECT_EQ(visitor.names[8], "state1");
}

/**
 * @brief Tests scoped traversal using different VisitorApplication scopes, limiting results to mapped nodes only.
 */
TEST(MappingGraph, ComplexGraph_OnlyMappedNodes)
{
    // Setup environment using helper function
    auto [root, inputs] = setupComplexGraphEnvironment();

    ASSERT_EQ(inputs.mappings.size(), 1);
    ASSERT_EQ(inputs.mechanicalStates.size(), 2);
    ASSERT_EQ(inputs.forceFields.size(), 4);
    sofa::simulation::MappingGraph mappingGraph(inputs);
    ASSERT_TRUE(mappingGraph.isBuilt());

    CollectNamesVisitor visitor;

    // Test ONLY_MAPPED_NODES scope
    mappingGraph.traverseTopDown(visitor, sofa::simulation::VisitorApplication::ONLY_MAPPED_NODES);
    ASSERT_GT(visitor.names.size(), 3);
    EXPECT_EQ(visitor.names[0], "state2");
    EXPECT_EQ(visitor.names[1], "ff2");
    EXPECT_EQ(visitor.names[2], "mass2");

}

/**
 * @brief Tests scoped traversal using different VisitorApplication scopes, limiting results to main nodes only.
 */
TEST(MappingGraph, ComplexGraph_OnlyMainNodes)
{
    // Setup environment using helper function
    auto [root, inputs] = setupComplexGraphEnvironment();

    ASSERT_EQ(inputs.mappings.size(), 1);
    ASSERT_EQ(inputs.mechanicalStates.size(), 2);
    ASSERT_EQ(inputs.forceFields.size(), 4);
    sofa::simulation::MappingGraph mappingGraph(inputs);
    ASSERT_TRUE(mappingGraph.isBuilt());

    CollectNamesVisitor visitor;

    // Test ONLY_MAIN_NODES scope
    mappingGraph.traverseTopDown(visitor, sofa::simulation::VisitorApplication::ONLY_MAIN_NODES);
    ASSERT_EQ(visitor.names.size(), 5);

    EXPECT_EQ(visitor.names[0], "state1");
    EXPECT_EQ(visitor.names[1], "ff1");
    EXPECT_EQ(visitor.names[2], "mass1");
    EXPECT_EQ(visitor.names[3], "mass1");
    EXPECT_EQ(visitor.names[4], "mapping");
}

}
