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

#include <SceneChecking/SceneCheckSpecialCharacters.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>

namespace sofa::scenechecking
{

const bool SceneCheckSpecialCharactersRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckSpecialCharacters::newSPtr());

const std::string SceneCheckSpecialCharacters::getName() { return "SceneCheckSpecialCharacters"; }

const std::string SceneCheckSpecialCharacters::getDesc()
{
    return "Check if nodes and components have special characters that may lead to undefined "
           "behavior.";
}

void SceneCheckSpecialCharacters::doInit(sofa::simulation::Node* node)
{
    SOFA_UNUSED(node);
    m_numWithSpecialChars = 0;
}

namespace
{

std::string containsSpecialCharacters(const std::string& str)
{
    static constexpr std::string_view specialChars = " !@#$%^&*()+=[]{}|;':\",./\\<>?";

    std::string foundChars {};

    for (const auto& c : specialChars)
    {
        if (str.find_first_of(c) != std::string::npos)
        {
            foundChars += c;
        }
    }

    return foundChars;
}

}

void SceneCheckSpecialCharacters::doCheckOn(sofa::simulation::Node* node)
{
    if (node == nullptr)
        return;

    if (const auto nodeSpecialChars = containsSpecialCharacters(node->getName());
        !nodeSpecialChars.empty())
    {
        msg_warning(node) << "The node has the following special characters in its name: '" << nodeSpecialChars << "'";
        ++m_numWithSpecialChars;
    }

    for (auto& object : node->object )
    {
        if (const sofa::core::objectmodel::BaseComponent* o = object.get())
        {
            if (const auto componentSpecialChars = containsSpecialCharacters(o->getName());
                !componentSpecialChars.empty())
            {
                msg_warning(o) << "The component has the following special characters in its name: '" << componentSpecialChars << "'";
                ++m_numWithSpecialChars;
            }
        }
    }
}

void SceneCheckSpecialCharacters::doPrintSummary()
{
    if (m_numWithSpecialChars != 0)
    {
        msg_warning(this->getName()) << "Found " << m_numWithSpecialChars
            << " nodes or components with special characters in their names. It can lead to undefined behavior.";
    }
}

}  // namespace sofa::scenechecking
