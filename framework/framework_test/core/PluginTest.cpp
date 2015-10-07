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
