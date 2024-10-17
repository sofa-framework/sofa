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

#include <string>
#include <sofa/core/config.h>
#include <sofa/core/fwd.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/DeprecatedData.h>

namespace sofa::core::objectmodel::lifecycle
{

DeprecatedData::DeprecatedData(Base* b, const std::string& deprecationVersion, const std::string& removalVersion, const std::string& name, const std::string& helptext)
{
    m_deprecationVersion = deprecationVersion;
    m_removalVersion = removalVersion;
    m_name = name;
    m_helptext = helptext;
    m_isRemoved = false;
    b->addDeprecatedAttribute(this);
}

} // namespace sofa
