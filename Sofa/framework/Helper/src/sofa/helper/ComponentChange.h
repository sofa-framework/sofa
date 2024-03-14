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
#include <vector>
#include <map>
#include <sstream>
#include <sofa/helper/config.h>

namespace sofa::helper::lifecycle
{

class SOFA_HELPER_API ComponentChange
{
public:
    ComponentChange() = default;
    explicit ComponentChange(const std::string& sinceVersion)
    {
        std::stringstream output;
        output << "This component changed since SOFA " << sinceVersion;
        m_message = output.str();
        m_changeVersion = sinceVersion;
    }
    virtual ~ComponentChange() = default;

    std::string m_message;
    std::string m_changeVersion;
    const std::string& getMessage() const { return m_message; }
    const std::string& getVersion() const { return m_changeVersion; }
};

class SOFA_HELPER_API Deprecated : public ComponentChange
{
public:
    explicit Deprecated(const std::string& sinceVersion, const std::string& untilVersion, const std::string& instruction="\b")
    {
        std::stringstream output;
        output << "This component has been DEPRECATED since SOFA " << sinceVersion << " "
                  "and will be removed in SOFA " << untilVersion << ". "
               << instruction <<
                  "\nPlease consider updating your scene as using "
                  "deprecated component may result in poor performance and undefined behavior. "
                  "If this component is crucial to you please report in a GitHub issue "
                  "in order to reconsider this component for future re-integration.";
        m_message = output.str();
        m_changeVersion = untilVersion;
    }
};

class SOFA_HELPER_API Pluginized : public ComponentChange
{
public:
    explicit Pluginized(const std::string& sinceVersion, const std::string& plugin)
    {
        std::stringstream output;
        output << "This component has been PLUGINIZED since SOFA " << sinceVersion << ". "
                  "To continue using this component you need to update you scene "
                  "and add <RequiredPlugin name='" <<  plugin << "'/>";
        m_message = output.str();
        m_changeVersion = sinceVersion;
    }
};

class SOFA_HELPER_API Removed : public ComponentChange
{
public:
    explicit Removed(const std::string&  sinceVersion, const std::string& atVersion)
    {
        std::stringstream output;
        output << "This component has been REMOVED since SOFA " << atVersion << " "
                  "(deprecated since " << sinceVersion << "). "
                  "\nPlease consider updating your scene. "
                  "If this component is crucial to you please report in a GitHub issue "
                  "in order to reconsider this component for future re-integration.";
        m_message = output.str();
        m_changeVersion = atVersion;
    }
};

class SOFA_HELPER_API Moved : public ComponentChange
{
public:
    Moved(const std::string& sinceVersion, const std::string& fromPlugin, const std::string& toPlugin)
    {
        std::stringstream output;
        output << "This component has been MOVED from " << fromPlugin << " to " << toPlugin << " since SOFA " << sinceVersion << ".\n"
            << "To continue using this component you may need to update your scene "
            << "by adding\n<RequiredPlugin name='" << toPlugin << "'/>";
        m_message = output.str();
        m_changeVersion = sinceVersion;
    }
};

class SOFA_HELPER_API Renamed : public ComponentChange
{
public:
    Renamed(const std::string& sinceVersion, const std::string& untilVersion,  const std::string& newName)
    {
        std::stringstream output;
        output << "This component has been RENAMED to " << newName  << " since SOFA " << sinceVersion
            << ", and this alias will be removed in SOFA " << untilVersion << "."
            << " To continue using this component after SOFA "<< untilVersion <<" you will need to update your scene ";
        m_message = output.str();
        m_changeVersion = untilVersion;
        m_newName = newName;
    }

   const std::string& getNewName() const
    {
        return m_newName;
    }

private:
    std::string m_newName;
};

extern SOFA_HELPER_API const std::map< std::string, Deprecated, std::less<> > deprecatedComponents;
extern SOFA_HELPER_API const std::map< std::string, ComponentChange, std::less<> > movedComponents;
extern SOFA_HELPER_API const std::map< std::string, Renamed, std::less<> > renamedComponents;
extern SOFA_HELPER_API const std::map< std::string, ComponentChange, std::less<> > uncreatableComponents;

} // namespace sofa::helper::lifecycle
