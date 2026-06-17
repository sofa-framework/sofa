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

#include <string_view>

namespace sofa::type
{

template<class T> struct TypeTrait{};

#define MAKE_TYPE_TRAIT(type, suffix_string) \
    template<> struct TypeTrait<type> \
    { \
        static constexpr std::string_view typeName = #type; \
        static constexpr std::string_view suffix = suffix_string; \
        static const std::string name() { return std::string(suffix); } \
        static const std::string GetTypeName() { return std::string(typeName); } \
    }

MAKE_TYPE_TRAIT(bool, "bool");

MAKE_TYPE_TRAIT(float, "f");
MAKE_TYPE_TRAIT(double, "d");
MAKE_TYPE_TRAIT(long double, "e");

MAKE_TYPE_TRAIT(int, "i");
MAKE_TYPE_TRAIT(unsigned int, "I");

MAKE_TYPE_TRAIT(short, "h");
MAKE_TYPE_TRAIT(unsigned short, "H");

MAKE_TYPE_TRAIT(char, "b");
MAKE_TYPE_TRAIT(unsigned char, "B");

MAKE_TYPE_TRAIT(long, "l");
MAKE_TYPE_TRAIT(unsigned long, "L");

MAKE_TYPE_TRAIT(long long, "q");
MAKE_TYPE_TRAIT(unsigned long long, "Q");

#undef MAKE_TYPE_TRAIT

}
