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
#include <sofa/testing/ScopedPlugin.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/core/ExecParams.h>
#include <sofa/component/playback/CompareState.h>
#include <sofa/component/playback/ReadState.h>
#include <sofa/component/playback/WriteState.h>
#include <sofa/component/playback/StateComponentLookup.h>

namespace
{

using sofa::simulation::Node;
using namespace sofa::component::playback;

/// A Node containing two mechanical states, simulated by an integration scheme
class StateCreators_test : public sofa::testing::BaseTest
{
protected:
    void doSetUp() override
    {
        m_plugins = sofa::testing::makeScopedPlugin({Sofa.Component.StateContainer, Sofa.Component.IntegrationScheme.Backward});
        m_root = sofa::simpleapi::createNode("root");
        sofa::simpleapi::createObject(m_root, "EulerImplicitIntegrationScheme");
        sofa::simpleapi::createObject(m_root, "MechanicalObject", {{"name", "A"}, {"size", "1"}});
        sofa::simpleapi::createObject(m_root, "MechanicalObject", {{"name", "B"}, {"size", "1"}});
        ASSERT_EQ(m_root->mechanicalState.size(), 2);
    }

    /// Check that there is exactly one component of type T per mechanical state, explicitly linked to it
    template<class T>
    void checkOneComponentPerState() const
    {
        const auto components = getLocalComponents<T>(m_root.get());
        ASSERT_EQ(components.size(), 2);
        EXPECT_EQ(components[0]->l_state.get(), m_root->mechanicalState[0]);
        EXPECT_EQ(components[1]->l_state.get(), m_root->mechanicalState[1]);
    }

    std::string sceneName() const
    {
        return std::string(SOFA_COMPONENT_PLAYBACK_TEST_BUILD_DIR) + "StateCreators_test";
    }

    std::unique_ptr<sofa::testing::ScopedPlugin> m_plugins;
    Node::SPtr m_root;
};

TEST_F(StateCreators_test, writeStateCreator_oneWriteStatePerState)
{
    WriteStateCreator creator(sofa::core::execparams::defaultInstance(), sceneName(), true, false, false, false);
    m_root->executeVisitor(&creator);
    checkOneComponentPerState<WriteState>();

    // running the creator again reuses the existing components
    m_root->executeVisitor(&creator);
    checkOneComponentPerState<WriteState>();
}

TEST_F(StateCreators_test, readStateCreator_oneReadStatePerState)
{
    ReadStateCreator creator(sceneName(), false, sofa::core::execparams::defaultInstance(), false);
    m_root->executeVisitor(&creator);
    checkOneComponentPerState<ReadState>();

    // running the creator again reuses the existing components
    m_root->executeVisitor(&creator);
    checkOneComponentPerState<ReadState>();
}

TEST_F(StateCreators_test, compareStateCreator_oneCompareStatePerState)
{
    CompareStateCreator creator(sceneName(), sofa::core::execparams::defaultInstance(), false);
    m_root->executeVisitor(&creator);
    checkOneComponentPerState<CompareState>();

    // running the creator again reuses the existing components
    m_root->executeVisitor(&creator);
    checkOneComponentPerState<CompareState>();
}

TEST_F(StateCreators_test, readStateCreator_reuseComponentLinkedToState)
{
    // a ReadState already exists, explicitly linked to the second state
    const auto existing = sofa::core::objectmodel::New<ReadState>();
    existing->l_state.set(m_root->mechanicalState[1]);
    m_root->addObject(existing);

    ReadStateCreator creator(sceneName(), false, sofa::core::execparams::defaultInstance(), false);
    m_root->executeVisitor(&creator);

    const auto components = getLocalComponents<ReadState>(m_root.get());
    ASSERT_EQ(components.size(), 2);
    EXPECT_EQ(components[0], existing.get());
    EXPECT_EQ(components[0]->l_state.get(), m_root->mechanicalState[1]);
    EXPECT_EQ(components[1]->l_state.get(), m_root->mechanicalState[0]);
}

TEST_F(StateCreators_test, readStateCreator_emptyLinkIsNotReusedWhenAmbiguous)
{
    // a ReadState already exists, without explicit link: with two states, it is ambiguous
    const auto existing = sofa::core::objectmodel::New<ReadState>();
    m_root->addObject(existing);

    ReadStateCreator creator(sceneName(), false, sofa::core::execparams::defaultInstance(), false);
    m_root->executeVisitor(&creator);

    const auto components = getLocalComponents<ReadState>(m_root.get());
    ASSERT_EQ(components.size(), 3);
    EXPECT_TRUE(existing->l_state.empty());
    EXPECT_EQ(components[1]->l_state.get(), m_root->mechanicalState[0]);
    EXPECT_EQ(components[2]->l_state.get(), m_root->mechanicalState[1]);
}

}
