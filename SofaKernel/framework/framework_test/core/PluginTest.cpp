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
#include <sofa/core/Plugin.h>
#include <gtest/gtest.h>

#include <sofa/core/objectmodel/BaseObject.h>

using namespace sofa::core;
using namespace sofa::core::objectmodel;

class Foo: public BaseObject {

};

template <class C>
class Bar: public BaseObject {

};

class TestPlugin : public sofa::core::Plugin {
public:
    TestPlugin(): sofa::core::Plugin("TestPlugin") {
        addComponent<Foo>("Component Foo");
        addComponent< Bar<float> >("Component Bar");
        addTemplateInstance< Bar<double> >();
    }
};


TEST(PluginTest, foo)
{
    TestPlugin plugin;
    EXPECT_EQ(2, plugin.getComponentEntries().size());
    ASSERT_TRUE(plugin.getComponentEntries().count("Foo") == 1);
    const Plugin::ComponentEntry foo = plugin.getComponentEntries().find("Foo")->second;
    EXPECT_EQ("Foo", foo.name);
    EXPECT_FALSE(foo.isATemplate);
    EXPECT_EQ("", foo.defaultTemplateParameters);
    EXPECT_TRUE(foo.aliases.empty());
    EXPECT_EQ(1, foo.creators.size());
    ASSERT_TRUE(plugin.getComponentEntries().count("Bar") == 1);
    const Plugin::ComponentEntry bar = plugin.getComponentEntries().find("Bar")->second;
    EXPECT_EQ("Bar", bar.name);
    EXPECT_TRUE(bar.isATemplate);
    EXPECT_EQ("float", bar.defaultTemplateParameters);
    EXPECT_TRUE(bar.aliases.empty());
    EXPECT_EQ(2, bar.creators.size());
    EXPECT_EQ(1, bar.creators.count("float"));
    EXPECT_EQ(1, bar.creators.count("double"));
}

// TEST_F(PluginTest, AddNonTemplatedComponent)
// {
//     plugin.addComponent<Foo>("Component Foo");
//     EXPECT_EQ(1, plugin.getComponentEntries().size());
//     ASSERT_TRUE(plugin.getComponentEntries().count("Foo") == 1);
//     const Plugin::ComponentEntry foo = plugin.getComponentEntries().find("Foo")->second;
//     EXPECT_EQ("Foo", foo.name);
//     EXPECT_FALSE(foo.isATemplate);
//     EXPECT_EQ("", foo.defaultTemplateParameters);
//     EXPECT_TRUE(foo.aliases.empty());
//     EXPECT_EQ(1, foo.creators.size());
// }

// TEST_F(PluginTest, AddTemplatedComponent)
// {
//     plugin.addComponent< Bar<float> >("Component Bar");
//     plugin.addTemplateInstance< Bar<double> >();
//     EXPECT_EQ(1, plugin.getComponentEntries().size());
//     ASSERT_TRUE(plugin.getComponentEntries().count("Bar") == 1);
//     const Plugin::ComponentEntry bar = plugin.getComponentEntries().find("Bar")->second;
//     EXPECT_EQ("Bar", bar.name);
//     EXPECT_TRUE(bar.isATemplate);
//     EXPECT_EQ("float", bar.defaultTemplateParameters);
//     EXPECT_TRUE(bar.aliases.empty());
//     EXPECT_EQ(2, bar.creators.size());
//     EXPECT_EQ(1, bar.creators.count("float"));
//     EXPECT_EQ(1, bar.creators.count("double"));
// }

// TEST_F(PluginTest, AddMultipleComponents)
// {
//     plugin.addComponent<Foo>("Component Foo");
//     plugin.addComponent< Bar<float> >("Component Bar");
//     plugin.addTemplateInstance< Bar<double> >();
//     EXPECT_EQ(2, plugin.getComponentEntries().size());
//     ASSERT_TRUE(plugin.getComponentEntries().count("Foo") == 1);
//     const Plugin::ComponentEntry foo = plugin.getComponentEntries().find("Foo")->second;
//     EXPECT_EQ("Foo", foo.name);
//     EXPECT_FALSE(foo.isATemplate);
//     EXPECT_EQ("", foo.defaultTemplateParameters);
//     EXPECT_TRUE(foo.aliases.empty());
//     EXPECT_EQ(1, foo.creators.size());
//     ASSERT_TRUE(plugin.getComponentEntries().count("Bar") == 1);
//     const Plugin::ComponentEntry bar = plugin.getComponentEntries().find("Bar")->second;
//     EXPECT_EQ("Bar", bar.name);
//     EXPECT_TRUE(bar.isATemplate);
//     EXPECT_EQ("float", bar.defaultTemplateParameters);
//     EXPECT_TRUE(bar.aliases.empty());
//     EXPECT_EQ(2, bar.creators.size());
//     EXPECT_EQ(1, bar.creators.count("float"));
//     EXPECT_EQ(1, bar.creators.count("double"));
// }
