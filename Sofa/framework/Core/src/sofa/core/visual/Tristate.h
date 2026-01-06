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

#include <sofa/core/config.h>

namespace sofa::core::visual
{

struct SOFA_CORE_API tristate
{
    enum state_t { false_value, true_value, neutral_value } state;
    tristate(bool b):state(b==true ? true_value : false_value )
    {
    }
    tristate():state(true_value)
    {
    }
    tristate(state_t s):state(s) {}

    operator bool() const
    {
        return state == true_value ? true : false;
    }

    bool operator==(const tristate& t) const
    {
        return state == t.state;
    }

    bool operator!=(const tristate& t) const
    {
        return state != t.state;
    }

    bool operator==(const state_t& s) const
    {
        return state == s;
    }

    bool operator!=(const state_t& s) const
    {
        return state != s;
    }

    friend inline tristate fusion_tristate(const tristate& lhs, const tristate& rhs);
    friend inline tristate merge_tristate(const tristate& previous, const tristate& current);
    friend inline tristate difference_tristate(const tristate& previous, const tristate& current);
};

inline tristate fusion_tristate(const tristate &lhs, const tristate &rhs)
{
    if( lhs.state == rhs.state ) return lhs;
    return tristate(tristate::neutral_value);
}

inline tristate merge_tristate(const tristate& previous, const tristate& current)
{
    if(current.state == tristate::neutral_value ) return previous;
    return current;
}
inline tristate difference_tristate(const tristate& previous, const tristate& current)
{
    if( current.state == tristate::neutral_value || current.state == previous.state )
        return tristate(tristate::neutral_value);
    return current;
}
}
