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
#include <sofa/type/vector.h>
#include <sofa/core/visual/Tristate.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <string>

namespace sofa::core::visual
{

class SOFA_CORE_API FlagTreeItem
{
protected:
    // Creating a case insensitive "find" function for map
    struct ci_comparison
    {
        // case-independent (ci) comparison
        struct nocase_compare
        {
            bool operator() (const unsigned char& c1, const unsigned char& c2) const
            {
                return tolower (c1) < tolower (c2);
            }
        };
        bool operator() (const std::string & s1, const std::string & s2) const
        {
            return std::lexicographical_compare(s1.begin (), s1.end (), s2.begin (), s2.end (), nocase_compare ());
        }
    };

    enum class READ_FLAG : char
    {
        KNOWN_FLAG,
        INCORRECT_LETTER_CASE,
        UNKNOWN_FLAG
    };

    type::vector<std::string> m_showName;
    sofa::type::vector<std::string> m_hideName;
    tristate m_state;

    FlagTreeItem* m_parent;
    sofa::type::vector<FlagTreeItem*> m_child;

    typedef type::vector<FlagTreeItem*>::iterator ChildIterator;
    typedef type::vector<FlagTreeItem*>::const_iterator ChildConstIterator;

public:
    FlagTreeItem(const std::string& showName, const std::string& hideName, FlagTreeItem* parent = nullptr);

    const tristate& state( ) const {return m_state;}
    tristate& state() {return m_state;}

    SOFA_CORE_API friend std::ostream& operator<< ( std::ostream& os, const FlagTreeItem& root );
    SOFA_CORE_API friend std::istream& operator>> ( std::istream& in, FlagTreeItem& root );
    std::ostream& write(std::ostream& os) const;
    std::istream& read(std::istream& in);
    std::istream& read(std::istream& in,
                       const std::function<void(std::string)>& unknownFlagFunction,
                       const std::function<void(std::string, std::string)>& incorrectLetterCaseFunction);

    void setValue(const tristate& state);

    void addAliasShow(const std::string& newAlias);
    void addAliasHide(const std::string& newAlias);
    void addAlias(sofa::type::vector<std::string> &name, const std::string &newAlias);

    void getLabels(sofa::type::vector<std::string>& labels) const;

protected:
    void propagateStateDown(FlagTreeItem* origin);
    void propagateStateUp(FlagTreeItem* origin);
    static std::map<std::string,bool, ci_comparison> create_flagmap(FlagTreeItem* root);
    static void create_parse_map(FlagTreeItem* root, std::map<std::string,bool,ci_comparison>& map);
    static void read_recursive(FlagTreeItem* root, const std::map<std::string,bool,ci_comparison>& map);
    static void write_recursive(const FlagTreeItem* root,  std::string& str);

    static READ_FLAG readFlag(std::map<std::string, bool, FlagTreeItem::ci_comparison>& parseMap,
                              std::string flag);
};

}
