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
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MappingGraph.h>
#include <sofa/testing/BaseTest.h>

namespace sofa
{

TEST(mappingGraphBreadthFirstTraversal, emptyGraph)
{
    auto root = simpleapi::createRootNode(simulation::getSimulation(), "root");
    sofa::type::vector<core::BaseMapping*> orderedMappings;

    sofa::simulation::mappingGraphBreadthFirstTraversal(root.get(), [&orderedMappings](core::BaseMapping* mapping)
    {
        orderedMappings.push_back(mapping);
    });

    EXPECT_TRUE(orderedMappings.empty());
}

TEST(mappingGraphBreadthFirstTraversal, oneMapping)
{
    const auto plugins = sofa::testing::makeScopedPlugin({Sofa.Component.StateContainer, Sofa.Component.Mapping.Linear});

    auto root = simpleapi::createRootNode(simulation::getSimulation(), "root");
    simpleapi::createObject(root, "MechanicalObject", {{"template", "Vec3"}, {"name", "dofs1"}});
    auto subnode = simpleapi::createChild(root, "subnode");
    simpleapi::createObject(subnode, "MechanicalObject", {{"template", "Vec3"}, {"name", "dofs2"}});
    simpleapi::createObject(subnode, "IdentityMapping", {{"template", "Vec3,Vec3"}, {"name", "mapping"},
        {"input", "@../dofs1"}, {"output", "@dofs2"}});

    sofa::type::vector<core::BaseMapping*> orderedMappings;

    sofa::simulation::mappingGraphBreadthFirstTraversal(root.get(), [&orderedMappings](core::BaseMapping* mapping)
    {
        orderedMappings.push_back(mapping);
    });

    EXPECT_EQ(orderedMappings.size(), 1);
}

TEST(mappingGraphBreadthFirstTraversal, twoIndependentMappings)
{
    const auto plugins = sofa::testing::makeScopedPlugin({Sofa.Component.StateContainer, Sofa.Component.Mapping.Linear});

    auto root = simpleapi::createRootNode(simulation::getSimulation(), "root");

    {
        auto subnode1 = simpleapi::createChild(root, "subnode1");
        simpleapi::createObject(subnode1, "MechanicalObject", {{"template", "Vec3"}, {"name", "dofs1"}});

        auto subsubnode1 = simpleapi::createChild(subnode1, "subsubnode1");
        simpleapi::createObject(subsubnode1, "MechanicalObject", {{"template", "Vec3"}, {"name", "dofs3"}});
        simpleapi::createObject(subsubnode1, "IdentityMapping", {{"template", "Vec3,Vec3"}, {"name", "mapping"},
            {"input", "@../dofs1"}, {"output", "@dofs3"}});
    }

    {
        auto subnode2 = simpleapi::createChild(root, "subnode2");
        simpleapi::createObject(subnode2, "MechanicalObject", {{"template", "Vec3"}, {"name", "dofs2"}});

        auto subsubnode2 = simpleapi::createChild(subnode2, "subsubnode2");
        simpleapi::createObject(subsubnode2, "MechanicalObject", {{"template", "Vec3"}, {"name", "dofs3"}});
        simpleapi::createObject(subsubnode2, "IdentityMapping", {{"template", "Vec3,Vec3"}, {"name", "mapping"},
            {"input", "@../dofs2"}, {"output", "@dofs3"}});
    }

    sofa::type::vector<core::BaseMapping*> orderedMappings;

    sofa::simulation::mappingGraphBreadthFirstTraversal(root.get(), [&orderedMappings](core::BaseMapping* mapping)
    {
        orderedMappings.push_back(mapping);
    });

    EXPECT_EQ(orderedMappings.size(), 2);
}

auto chainedMappings(simulation::MappingGraphDirection direction)
{
    const auto plugins = sofa::testing::makeScopedPlugin({Sofa.Component.StateContainer, Sofa.Component.Mapping.Linear});

    auto root = simpleapi::createRootNode(simulation::getSimulation(), "root");
    simpleapi::createObject(root, "MechanicalObject", {{"template", "Vec3"}, {"name", "dofs1"}});

    auto subnode1 = simpleapi::createChild(root, "subnode1");
    simpleapi::createObject(subnode1, "MechanicalObject",
                            {{"template", "Vec3"}, {"name", "dofs2"}});

    auto mapping1 = simpleapi::createObject(subnode1, "IdentityMapping",
                            {{"template", "Vec3,Vec3"},
                             {"name", "mapping"},
                             {"input", "@../dofs1"},
                             {"output", "@dofs2"}});

    auto subsubnode1 = simpleapi::createChild(subnode1, "subsubnode1");
    simpleapi::createObject(subsubnode1, "MechanicalObject",
                            {{"template", "Vec3"}, {"name", "dofs3"}});
    auto mapping2 = simpleapi::createObject(subsubnode1, "IdentityMapping",
                            {{"template", "Vec3,Vec3"},
                             {"name", "mapping"},
                             {"input", "@../dofs2"},
                             {"output", "@dofs3"}});

    sofa::type::vector<core::BaseMapping*> orderedMappings;

    sofa::simulation::mappingGraphBreadthFirstTraversal(root.get(), [&orderedMappings](core::BaseMapping* mapping)
    {
        orderedMappings.push_back(mapping);
    }, true, direction);

    return std::make_tuple(orderedMappings, mapping1, mapping2);
}

TEST(mappingGraphBreadthFirstTraversal, twoChainedMappingsTopDown)
{
    const auto [orderedMappings, mapping1, mapping2] =
        chainedMappings(simulation::MappingGraphDirection::TOP_DOWN);

    ASSERT_EQ(orderedMappings.size(), 2);
    EXPECT_EQ(orderedMappings[0], mapping1);
    EXPECT_EQ(orderedMappings[1], mapping2);
}

TEST(mappingGraphBreadthFirstTraversal, twoChainedMappingsBottomUp)
{
    const auto [orderedMappings, mapping1, mapping2] =
        chainedMappings(simulation::MappingGraphDirection::BOTTOM_UP);

    ASSERT_EQ(orderedMappings.size(), 2);
    EXPECT_EQ(orderedMappings[0], mapping2);
    EXPECT_EQ(orderedMappings[1], mapping1);
}

}
