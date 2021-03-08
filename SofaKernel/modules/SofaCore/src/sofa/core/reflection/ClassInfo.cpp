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
#include "ClassInfo.h"

namespace sofa::core::reflection
{

/// returns true iff c is a parent class of this
bool ClassInfo::hasParent(const ClassInfo* c) const
{
    if (this == c)
        return true;

    for (unsigned int i=0; i<parents.size(); ++i)
    {
        if (parents[i]->hasParent(c))
            return true;
    }
    return false;
}

/// returns true iff a parent class of this is named parentClassName
bool ClassInfo::hasParent(const std::string& parentClassName) const
{
    if (className==parentClassName)
        return true;
    for (unsigned int i=0; i<parents.size(); ++i)
    {
        if (parents[i]->hasParent(parentClassName))
            return true;
    }
    return false;
}

}
