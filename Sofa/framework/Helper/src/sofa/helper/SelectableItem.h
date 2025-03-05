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
#pragma once
#include <ostream>
#include <string>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/logging/ComponentInfo.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/type/fixed_array.h>
#include <map>


namespace sofa::helper
{

struct Item
{
    std::string_view key;
    std::string_view description;
};

/**
 * Type used to define the replacement key of a deprecated key
 */
struct DeprecatedItem
{
    std::string_view replacementKey;
    std::string_view deprecationMessage;
};


#define MAKE_SELECTABLE_ITEMS_BEGIN(ClassName, ...) \
    struct ClassName final : sofa::helper::SelectableItem<ClassName> \
    { \
        using sofa::helper::SelectableItem<ClassName>::SelectableItem; \
        using sofa::helper::SelectableItem<ClassName>::operator=; \
        using sofa::helper::SelectableItem<ClassName>::operator==; \
        static constexpr std::array s_items { \
            __VA_ARGS__ \
        }; \
        static_assert(std::is_same_v<decltype(s_items)::value_type, sofa::helper::Item>); \

/**
 * Helper macro to ease the creation of a class derived from SelectableItem
 *
 * @param ClassName Name to give to the class derived from SelectableItem
 *
 * Example:
 * MAKE_SELECTABLE_ITEMS(ResolutionMethod,
 *      sofa::helper::Item{"ProjectedGaussSeidel", "Projected Gauss-Seidel"},
 *      sofa::helper::Item{"UnbuiltGaussSeidel", "Gauss-Seidel no matrix assembly"},
 *      sofa::helper::Item{"NonsmoothNonlinearConjugateGradient", "Non-smooth non-linear conjugate gradient"}
 *  );
 */
#define MAKE_SELECTABLE_ITEMS(ClassName, ...) \
    MAKE_SELECTABLE_ITEMS_BEGIN(ClassName, __VA_ARGS__) }

/**
 * Version of the helper macro where a deprecation map is declared. Warning: It is not defined
 */
#define MAKE_SELECTABLE_ITEMS_WITH_DEPRECATION(ClassName, ...) \
    MAKE_SELECTABLE_ITEMS_BEGIN(ClassName, __VA_ARGS__) \
    static const std::map<std::string_view, sofa::helper::DeprecatedItem> s_deprecationMap; \
    }

struct BaseSelectableItem
{
    [[nodiscard]] virtual std::size_t getNumberOfItems() const { return 0; }
    [[nodiscard]] virtual const Item* getItemsData() const { return nullptr; }
    [[nodiscard]] virtual std::size_t getSelectedId() const { return 0; }
    virtual void setSelectedId(std::size_t /* id */) {}
    virtual void print(std::ostream& /* out */) const {}
    virtual void readFromStream(std::istream& /* stream */) {}
};

inline std::ostream& operator<<(std::ostream& out, const BaseSelectableItem& selectableItem)
{
    selectableItem.print(out);
    return out;
}

inline std::istream& operator>>(std::istream& in, BaseSelectableItem& selectableItem)
{
    selectableItem.readFromStream(in);
    return in;
}

template<class Derived, typename IdType>
class SelectableItem;

namespace logging
{

template<class Derived, typename IdType>
bool notMuted(const SelectableItem<Derived, IdType>* );

template<class Derived, typename IdType>
ComponentInfo::SPtr getComponentInfo(const SelectableItem<Derived, IdType>* );

}

/**
 * Selection of an item among a fixed list of items.
 *
 * The class is designed to have the list of items static, so the class can be
 * used in a constexpr context. In particular, it can be used in a switch
 * statement.
 *
 * The helper macro @MAKE_SELECTABLE_ITEMS can be used to make the code more concise.
 *
 * @tparam Derived The class derived from SelectableItem. The CRTP must be used.
 * @tparam IdType A type for the indices
 */
template<class Derived, typename IdType = std::size_t>
class SelectableItem : public BaseSelectableItem
{
public:

    static constexpr std::size_t numberOfItems()
    {
        return std::tuple_size<decltype(Derived::s_items)>{};
    }

    static constexpr const auto &items()
    {
        return Derived::s_items;
    }

    using id_type = IdType;

    constexpr SelectableItem() = default;
    constexpr SelectableItem(const std::string_view key) : m_selected_id(findId(key))
    {
        if (m_selected_id >= numberOfItems())
        {
            keyError(key);
        }
    }

    [[nodiscard]] std::string_view key() const
    {
        return Derived::s_items[m_selected_id].key;
    }

    [[nodiscard]] std::string_view description() const
    {
        return Derived::s_items[m_selected_id].description;
    }

    explicit operator std::string_view() const
    {
        return key();
    }

    constexpr operator id_type() const
    {
        return m_selected_id;
    }

    [[nodiscard]] bool operator==(const std::string_view key)
    {
        return key == Derived::s_items[m_selected_id].key;
    }

    [[nodiscard]] bool operator==(const SelectableItem& other) const
    {
        return m_selected_id == other.m_selected_id;
    }

    [[nodiscard]] bool operator!=(const SelectableItem& other) const
    {
        return m_selected_id != other.m_selected_id;
    }

    constexpr SelectableItem& operator=(const std::string_view key)
    {
        m_selected_id = findId(key);
        if (m_selected_id >= numberOfItems())
        {
            keyError(key);
        }
        return *this;
    }

    [[nodiscard]] static const std::string& allKeysAsString()
    {
        static std::string allKeys = ::sofa::helper::join(Derived::s_items.begin(), Derived::s_items.end(),
            [](const Item& item){ return item.key;}, ',');
        return allKeys;
    }

    static const std::string& dataDescription()
    {
        static const std::string dataDescription =
            sofa::helper::join(Derived::s_items.begin(), Derived::s_items.end(),
            [](const Item& item)
            {
                return "- " + std::string{item.key} + ": " + std::string{item.description};
            }, '\n');
        return dataDescription;
    }

    void print(std::ostream& out) const final
    {
        out << key();
    }

    void readFromStream(std::istream& stream) final
    {
        std::string tmp;
        std::getline(stream, tmp);
        m_selected_id = findId(tmp);
        if (m_selected_id >= numberOfItems())
        {
            keyError(tmp);
        }
    }

protected:
    id_type m_selected_id {};

    static constexpr id_type findId(const std::string_view key)
    {
        //use std::ranges::enumerate in C++23
        for (id_type i = 0; i < numberOfItems(); ++i)
        {
            if (key == Derived::s_items[i].key)
            {
                return i;
            }
        }
        return static_cast<id_type>(-1);
    }

    void keyError(const std::string_view key)
    {
        if constexpr (requires {Derived::s_deprecationMap;})
        {
            static_assert(std::is_same_v<std::remove_cv_t<decltype(Derived::s_deprecationMap)>, std::map<std::string_view, DeprecatedItem>>);
            const auto it = Derived::s_deprecationMap.find(key);
            if (it != Derived::s_deprecationMap.end())
            {
                if (it->second.replacementKey == key)
                {
                    dmsg_fatal() << "Item '" << key << "' is deprecated, but new key is also '" << it->second.replacementKey << "'.";
                }
                else
                {
                    msg_warning() << "Item '" << key << "' is deprecated. New key to use instead is '" << it->second.replacementKey << "'. " << it->second.deprecationMessage;
                    this->operator=(it->second.replacementKey);
                    return;
                }
            }
        }

        static_assert(numberOfItems() > 0);
        msg_error() << "Item '" << key << "' does not exist in the list ["
                << this->allKeysAsString() << "]. Fall back to item '"
                << items().front().key << "'";
        m_selected_id = 0;
    }

private:
    [[nodiscard]] std::size_t getNumberOfItems() const final
    {
        return numberOfItems();
    }

    [[nodiscard]] const Item* getItemsData() const final
    {
        return Derived::s_items.data();
    }

    [[nodiscard]] std::size_t getSelectedId() const final
    {
        return m_selected_id;
    }

    void setSelectedId(std::size_t id) final
    {
        m_selected_id = id;
    }
};

namespace logging
{

template<class Derived, typename IdType>
bool notMuted(const SelectableItem<Derived, IdType>* )
{
    return true;
}

template<class Derived, typename IdType>
ComponentInfo::SPtr getComponentInfo(const SelectableItem<Derived, IdType>* )
{
    return std::make_shared<ComponentInfo>("SelectableItem");
}
}

}

