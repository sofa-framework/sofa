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
#include <string>
#include <unordered_map>

namespace sofa::core
{

/**
 * Helper class to name a component based on its type.
 *
 * Two conventions are available for legacy reasons:
 * - XML: use a counter to add a unique suffix at the end of the name
 * - Python: the short name of the type is returned
 */
class SOFA_CORE_API ComponentNameHelper
{
public:

    enum class Convention : char
    {
        xml,
        python
    };

    std::string resolveName(const std::string& type, const std::string& name, Convention convention);
    std::string resolveName(const std::string& type, Convention convention);

private:

    std::unordered_map<std::string, unsigned int> m_instanceCounter;
};

}
