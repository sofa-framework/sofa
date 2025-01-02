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
#include <sofa/helper/SelectableItem.h>
#include <gtest/gtest.h>
#include <sofa/testing/TestMessageHandler.h>
#include <sofa/core/objectmodel/Data.h>


struct TestSelectableItem final : sofa::helper::SelectableItem<TestSelectableItem>
{
    using Inherited = sofa::helper::SelectableItem<TestSelectableItem>;
    using Inherited::SelectableItem;
    using Inherited::operator=;

    static constexpr std::array s_items {
        sofa::helper::Item{"small", "small displacements"},
        sofa::helper::Item{"large", "large displacements"},
    };

    inline static const std::map<std::string_view, sofa::helper::DeprecatedItem> s_deprecationMap {
        {"deprecated_key", sofa::helper::DeprecatedItem{"large", "Now use 'large'"}}
    };

    [[nodiscard]] id_type selectedId() const
    {
        return m_selected_id;
    }
};

MAKE_SELECTABLE_ITEMS(TestSelectableItemNoDeprecation,
    sofa::helper::Item{"small", "small displacements"},
    sofa::helper::Item{"large", "large displacements"});

MAKE_SELECTABLE_ITEMS_WITH_DEPRECATION(TestSelectableItemWithDeprecation,
    sofa::helper::Item{"small", "small displacements"},
    sofa::helper::Item{"large", "large displacements"});

const std::map<std::string_view, sofa::helper::DeprecatedItem> TestSelectableItemWithDeprecation::s_deprecationMap{
    {"deprecated_key", sofa::helper::DeprecatedItem{"large", "Now use 'large'"}}
};

TEST(SelectableItem, numberOfItems)
{
    static constexpr auto size = TestSelectableItem::numberOfItems();
    EXPECT_EQ(size, 2);
}

TEST(SelectableItem, allKeysAsString)
{
    static const auto& allKeys = TestSelectableItem::allKeysAsString();
    EXPECT_EQ(allKeys, "small,large");
}

TEST(SelectableItem, operator_equal_string_view)
{
    static constexpr std::string_view key_small = "small";

    TestSelectableItem foo;
    foo = key_small;

    EXPECT_EQ(foo.selectedId(), 0);
}

TEST(SelectableItem, operator_equal_string)
{
    static std::string key_large = "large";

    TestSelectableItem foo;
    foo = key_large;

    EXPECT_EQ(foo.selectedId(), 1);
}

TEST(SelectableItem, description)
{
    TestSelectableItem foo;
    foo = "large";

    EXPECT_EQ(foo.description(), "large displacements");
}

TEST(SelectableItem, operator_equal_to)
{
    TestSelectableItem foo;
    foo = "large";

    const bool is_equal = (foo == "large");
    EXPECT_TRUE(is_equal);
}

TEST(SelectableItem, convertion_string_view)
{
    TestSelectableItem foo;
    foo = "large";

    const auto key = static_cast<std::string_view>(foo);
    EXPECT_EQ(key, "large");
}

TEST(SelectableItem, stream_out)
{
    TestSelectableItem foo;
    foo = "large";

    std::stringstream ss;
    ss << foo;

    EXPECT_EQ(ss.str(), "large");
}

TEST(SelectableItem, stream_in)
{
    TestSelectableItem foo;

    std::istringstream input_stream("large");
    input_stream >> foo;

    EXPECT_EQ(foo.selectedId(), 1);
}

TEST(SelectableItem, constexpr_constructor)
{
    static constexpr TestSelectableItem foo("large");
    EXPECT_EQ(foo.selectedId(), 1);

    // static constexpr TestSelectableItem bar("bar"); //does not compile because "bar" does not exist in the list
}

TEST(SelectableItem, wrong_key)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    TestSelectableItem foo;

    {
        EXPECT_MSG_EMIT(Error);
        foo = "foo";
    }

    EXPECT_EQ(foo.selectedId(), 0);
}

TEST(SelectableItem, data)
{
    const sofa::Data<TestSelectableItem> data;
    EXPECT_EQ(data.getValueTypeString(), "SelectableItem");
}

TEST(SelectableItem, DataTypeInfo_BaseSelectableItems)
{
    EXPECT_EQ(sofa::defaulttype::DataTypeInfo<sofa::helper::BaseSelectableItem>::GetTypeName(), "SelectableItem");
}

TEST(SelectableItem, DataTypeInfoValidInfo_SelectableItem)
{
    EXPECT_TRUE(sofa::defaulttype::DataTypeInfo<TestSelectableItem>::ValidInfo);
}

TEST(SelectableItem, DataTypeInfo_SelectableItem)
{
    EXPECT_EQ(sofa::defaulttype::DataTypeInfo<TestSelectableItem>::GetTypeName(), "SelectableItem");
}

TEST(SelectableItem, BaseData_typeName)
{
    EXPECT_EQ(sofa::core::objectmodel::BaseData::typeName<TestSelectableItem>(), "SelectableItem");
}

TEST(SelectableItem, dynamic_cast_base)
{
    TestSelectableItem foo;
    EXPECT_NE(nullptr, dynamic_cast<sofa::helper::BaseSelectableItem*>(&foo));
}

template<class T>
void testDeprecatedKey()
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    T foo;

    EXPECT_MSG_EMIT(Warning);
    foo = "deprecated_key";

    const auto key = static_cast<std::string_view>(foo);
    EXPECT_EQ(key, "large");
}

TEST(SelectableItem, deprecated_key)
{
    testDeprecatedKey<TestSelectableItem>();
    testDeprecatedKey<TestSelectableItemWithDeprecation>();
}
