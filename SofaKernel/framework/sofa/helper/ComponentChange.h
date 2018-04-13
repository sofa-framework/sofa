/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_COMPONENTCHANGE_H
#define SOFA_HELPER_COMPONENTCHANGE_H

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "helper.h"

namespace sofa
{
namespace helper
{
namespace lifecycle
{

class SOFA_HELPER_API ComponentChange
{
public:
    std::string m_sinceVersion;

    ComponentChange(std::string sinceVersion)
        : m_sinceVersion(sinceVersion) {}

    virtual std::string getMessage()
    {
        std::stringstream output;
        output << "This component changed since SOFA " << m_sinceVersion;
        return output.str();
    }
};

class SOFA_HELPER_API Deprecated : public ComponentChange
{
public:
    std::string m_untilVersion;

    Deprecated(std::string sinceVersion, std::string untilVersion)
        : ComponentChange(sinceVersion), m_untilVersion(untilVersion) {}

    std::string getMessage()
    {
        std::stringstream output;
        output << "This component has been deprecated since SOFA " << m_sinceVersion << " "
                  "and will be removed in SOFA " << m_untilVersion << ". "
                  "Please consider updating your scene as using "
                  "deprecated component may result in poor performance and undefined behavior. "
                  "If this component is crucial to you please report that to sofa-dev@ so we can  "
                  "reconsider this component for future re-integration.";
        return output.str();
    }
};

class SOFA_HELPER_API Pluginized : public ComponentChange
{
public:
    std::string m_plugin;

    Pluginized(std::string sinceVersion, std::string plugin)
        : ComponentChange(sinceVersion), m_plugin(plugin) {}

    std::string getMessage()
    {
        std::stringstream output;
        output << "This component has been pluginized to " << m_plugin << " "
                  "since SOFA " << m_sinceVersion << ". "
                  "To continue using this component you need to update you scene "
                  "and add <RequiredPlugin name='" <<  m_plugin << "'/>";
        return output.str();
    }
};

class SOFA_HELPER_API Removed : public ComponentChange
{
public:
    Removed(std::string sinceVersion)
        : ComponentChange(sinceVersion) {}

    std::string getMessage()
    {
        std::stringstream output;
        output << "This component has been removed since SOFA " << m_sinceVersion << ". "
                  "Please consider updating your scene. "
                  "If this component is crucial to you please report that to sofa-dev@ so that we can "
                  "reconsider this component for future re-integration.";
        return output.str();
    }
};

extern SOFA_HELPER_API std::map< std::string, Deprecated > deprecatedComponents;
extern SOFA_HELPER_API std::map< std::string, ComponentChange > uncreatableComponents;

} // namespace lifecycle
} // namespace helper
} // namespace sofa

#endif
