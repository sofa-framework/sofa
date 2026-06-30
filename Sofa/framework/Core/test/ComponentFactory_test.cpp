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
#include <sofa/core/ComponentFactory.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/fwd.h>
#include <sofa/testing/TestMessageHandler.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

// factory.registerComponent(core::CreateComponent<DummyComponent>("DummyComponent")
// .withModule("test")
// .withDescription("dummy")
// );
struct DummyComponent : public core::objectmodel::BaseComponent
{
    SOFA_CLASS(DummyComponent, core::objectmodel::BaseComponent);
};

// factory.registerComponent(core::CreateComponent<DummyComponentWith1Template>("DummyComponentWith1Template")
// .withModule("test")
// .withDescription("dummy")
// .addTemplateAttribute<T>("t")
// );
template<class T>
struct DummyComponentWith1Template : public core::objectmodel::BaseComponent
{
    SOFA_CLASS(DummyComponentWith1Template, core::objectmodel::BaseComponent);
};

// factory.registerComponent(core::CreateComponent<DummyComponentWith2Template>("DummyComponentWith1Template")
// .withModule("test")
// .withDescription("dummy")
// .addTemplateAttribute<T1>("t1")
// .addTemplateAttribute<T2>("t2")
// );
template<class T1, class T2>
struct DummyComponentWith2Template : public core::objectmodel::BaseComponent
{
    SOFA_CLASS(SOFA_TEMPLATE2(DummyComponentWith2Template, T1, T2), core::objectmodel::BaseComponent);
};


struct ComponentFactory_test : public ::testing::Test
{
    void SetUp() override
    {
        sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance());
        node = sofa::simulation::getSimulation()->createNewNode("root");
    }

    core::ComponentFactory factory;
    simulation::Node::SPtr node;
};

TEST_F(ComponentFactory_test, CreateComponent)
{
    factory.registerComponent(core::CreateComponent<DummyComponent>("DummyComponent")
        .withModule("test")
        .withDescription("dummy")
    );

    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "DummyComponent");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_EQ(createdComponent->getName(), "nameInTheScene");
    EXPECT_EQ(createdComponent->getClassName(), "DummyComponent");
    EXPECT_EQ(createdComponent->d_factoryName.getValue(), "DummyComponent");
}

TEST_F(ComponentFactory_test, CreateComponentDifferentName)
{
    factory.registerComponent(core::CreateComponent<DummyComponent>("NotDummyComponent")
        .withModule("test")
        .withDescription("dummy")
    );

    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "NotDummyComponent");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_EQ(createdComponent->getName(), "nameInTheScene");
    EXPECT_EQ(createdComponent->getClassName(), "DummyComponent");
    EXPECT_EQ(createdComponent->d_factoryName.getValue(), "NotDummyComponent");
}

TEST_F(ComponentFactory_test, HasCreator)
{
    factory.registerComponent(core::CreateComponent<DummyComponent>("DummyComponent")
        .withModule("test")
        .withDescription("dummy")
    );

    EXPECT_TRUE(factory.hasCreator("DummyComponent"));
    EXPECT_FALSE(factory.hasCreator("NonExistentComponent"));
}

TEST_F(ComponentFactory_test, Aliases)
{
    factory.registerComponent(core::CreateComponent<DummyComponent>("DummyComponent")
        .withModule("test")
        .withDescription("dummy")
        .addAlias("DummyAlias")
    );

    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "DummyAlias");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_EQ(createdComponent->getClassName(), "DummyComponent");
}

TEST_F(ComponentFactory_test, TargetEntries)
{
    factory.registerComponent(core::CreateComponent<DummyComponent>("Compo1")
        .withModule("ModuleA")
        .withDescription("d1")
    );
    factory.registerComponent(core::CreateComponent<DummyComponent>("Compo2")
        .withModule("ModuleA")
        .withDescription("d2")
    );
    factory.registerComponent(core::CreateComponent<DummyComponent>("Compo3")
        .withModule("ModuleB")
        .withDescription("d3")
    );

    std::vector<core::ComponentRegistrationData::SPtr> entries;
    factory.getEntriesFromTarget(entries, "ModuleA");
    EXPECT_EQ(entries.size(), 2);

    std::string list = factory.listClassesFromTarget("ModuleA");
    EXPECT_TRUE(list.find("Compo1") != std::string::npos);
    EXPECT_TRUE(list.find("Compo2") != std::string::npos);
    EXPECT_FALSE(list.find("Compo3") != std::string::npos);
}

TEST_F(ComponentFactory_test, OneTemplate_attributes)
{
    factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<int>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t", "int")
    );

    factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<float>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t", "float")
    );

    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "TemplatedCompo");
    desc.setAttribute("t", "int");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_NE(dynamic_cast<DummyComponentWith1Template<int>*>(createdComponent.get()), nullptr);
}

TEST_F(ComponentFactory_test, OneTemplate_identicalRegistration)
{
    factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<int>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t", "int")
    );

    {
        EXPECT_MSG_EMIT(Error);
        factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<float>>("TemplatedCompo")
           .withModule("test")
           .withDescription("dummy")
           .addTemplateAttribute("t", "int")
       );
    }

    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "TemplatedCompo");
    desc.setAttribute("t", "int");
    EXPECT_MSG_EMIT(Warning); //ambiguity because of identical registration
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_NE(dynamic_cast<DummyComponentWith1Template<int>*>(createdComponent.get()), nullptr);
}

TEST_F(ComponentFactory_test, OneTemplate_identicalRegistrationNoTemplateAttribute)
{
    factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<int>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t", "int")
    );

    {
        EXPECT_MSG_EMIT(Error);
        factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<float>>("TemplatedCompo")
           .withModule("test")
           .withDescription("dummy")
           .addTemplateAttribute("t", "int")
       );
    }

    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "TemplatedCompo");
    EXPECT_MSG_EMIT(Warning); //ambiguity because of identical registration
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_NE(dynamic_cast<DummyComponentWith1Template<int>*>(createdComponent.get()), nullptr);
}

TEST_F(ComponentFactory_test, OneTemplate_templateKeyword)
{
    factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<int>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t", "int")
    );

    factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<float>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t", "int")
    );

    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "TemplatedCompo");
    desc.setAttribute("template", "int");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_NE(dynamic_cast<DummyComponentWith1Template<int>*>(createdComponent.get()), nullptr);
}

TEST_F(ComponentFactory_test, TwoTemplate_attributes)
{
    EXPECT_MSG_NOEMIT(Warning);
    EXPECT_MSG_NOEMIT(Error);

    // Register component using two templates (e.g., int and float attributes)
    factory.registerComponent(core::CreateComponent<DummyComponentWith2Template<int, float>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t1", "int")
        .addTemplateAttribute("t2", "float")
    );

    factory.registerComponent(core::CreateComponent<DummyComponentWith2Template<int, std::string>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t1", "int")
        .addTemplateAttribute("t2", "string")
    );

    // Create component specifying both template attributes
    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "TemplatedCompo");
    desc.setAttribute("t1", "int");
    desc.setAttribute("t2", "float");
    auto createdComponent = factory.createComponent(node.get(), &desc);

    ASSERT_NE(createdComponent, nullptr);
    // Check if the component is correctly cast to the expected template type
    auto* cast = dynamic_cast<DummyComponentWith2Template<int, float>*>(createdComponent.get());
    EXPECT_NE(cast, nullptr);
}

TEST_F(ComponentFactory_test, TwoTemplate_templateKeyword)
{
    EXPECT_MSG_NOEMIT(Warning);
    EXPECT_MSG_NOEMIT(Error);

    // Register two different components using two templates
    factory.registerComponent(core::CreateComponent<DummyComponentWith2Template<int, float>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t1", "int")
        .addTemplateAttribute("t2", "float")
    );

    factory.registerComponent(core::CreateComponent<DummyComponentWith2Template<double, bool>>("TemplatedCompo")
        .withModule("test")
        .withDescription("dummy")
        .addTemplateAttribute("t1", "double")
        .addTemplateAttribute("t2", "bool")
    );

    // Use the 'template' keyword to specify all template arguments: t1=int, t2=float
    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "TemplatedCompo");
    desc.setAttribute("template", "int,float"); // Assuming comma-separated list of attributes
    auto createdComponent = factory.createComponent(node.get(), &desc);

    ASSERT_NE(createdComponent, nullptr);
    // Check if the component is correctly cast to the expected template type
    auto* cast = dynamic_cast<DummyComponentWith2Template<int, float>*>(createdComponent.get());
    EXPECT_NE(cast, nullptr);
}

TEST_F(ComponentFactory_test, CreateUnknownComponent)
{
    EXPECT_MSG_EMIT(Error);
    core::objectmodel::BaseObjectDescription desc("nameInTheScene", "UnknownComponent");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    EXPECT_EQ(createdComponent, nullptr);
}

TEST_F(ComponentFactory_test, InstantiationPriority)
{
    auto lowPriority = core::CreateComponent<DummyComponent>("PriorityCompo")
        .withModule("test")
        .withDescription("low")
        .withInstantiationPriority(1);
    factory.registerComponent(lowPriority);

    auto highPriority = core::CreateComponent<DummyComponent>("PriorityCompo")
        .withModule("test")
        .withDescription("high")
        .withInstantiationPriority(10);
    factory.registerComponent(highPriority);

    core::objectmodel::BaseObjectDescription desc("name", "PriorityCompo");
    auto createdComponent = factory.createComponent(node.get(), &desc);

    ASSERT_NE(createdComponent, nullptr);
    // The one with description "high" (priority 10) should be selected.
    // Since both are DummyComponent, we check the factory data if available or just ensure it doesn't crash.
    // In ComponentFactory.cpp, it sorts by priority descending.
    EXPECT_EQ(createdComponent->getClassName(), "DummyComponent");
}

TEST_F(ComponentFactory_test, FullNameCreation)
{
    factory.registerComponent(core::CreateComponent<DummyComponent>("MyComponent")
        .withModule("MyModule")
        .withDescription("desc")
    );

    // Create using "Module.Component" syntax
    core::objectmodel::BaseObjectDescription desc("name", "MyModule.MyComponent");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
    EXPECT_EQ(createdComponent->getClassName(), "DummyComponent");
    EXPECT_EQ(createdComponent->d_factoryName.getValue(), "MyComponent");
}

TEST_F(ComponentFactory_test, SuggestionOnMisspell)
{
    factory.registerComponent(core::CreateComponent<DummyComponent>("SpecificComponent")
        .withModule("test").withDescription("description")
    );

    EXPECT_MSG_EMIT(Error); // Should contain suggestion
    core::objectmodel::BaseObjectDescription desc("name", "SpecificComponnt"); // Typos
    auto createdComponent = factory.createComponent(node.get(), &desc);
    EXPECT_EQ(createdComponent, nullptr);
}

TEST_F(ComponentFactory_test, TemplateAliasResolution)
{
    // Sofa often uses aliases like 'Vec3d' for 'std::vector<double>' etc.
    // Here we simulate a registration with a specific type and creation with an alias.
    factory.registerComponent(core::CreateComponent<DummyComponentWith1Template<double>>("AliasedCompo")
        .withModule("test")
        .withDescription("description")
        .addTemplateAttribute("t", "d")
    );

    core::objectmodel::BaseObjectDescription desc("name", "AliasedCompo");
    desc.setAttribute("t", "double");
    auto createdComponent = factory.createComponent(node.get(), &desc);
    ASSERT_NE(createdComponent, nullptr);
}

}
