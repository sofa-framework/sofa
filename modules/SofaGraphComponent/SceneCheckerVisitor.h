/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_SCENECHECKERVISTOR_H
#define SOFA_SIMULATION_SCENECHECKERVISTOR_H

#include "config.h"

#include <functional>
#include <map>

#include <sofa/simulation/Visitor.h>

namespace sofa
{
namespace simulation
{
typedef std::function<void(sofa::core::objectmodel::Base*)> ChangeSetHookFunction ;

class SOFA_GRAPH_COMPONENT_API SceneCheckerVisitor : public Visitor
{
public:
    SceneCheckerVisitor(const sofa::core::ExecParams* params) ;
    virtual ~SceneCheckerVisitor() ;

    void validate(Node* node) ;

    void enableValidationAPIVersion(Node *node) ;
    void enableValidationRequiredPlugins(Node* node) ;

    virtual Result processNodeTopDown(Node* node) override ;

    void installChangeSets() ;
    void addHookInChangeSet(const std::string& version, ChangeSetHookFunction fct) ;
private:
    std::map<std::string,bool> m_requiredPlugins ;
    bool m_isRequiredPluginValidationEnabled {true} ;
    bool m_isAPIVersionValidationEnabled {true} ;
    std::string m_currentApiLevel;
    std::string m_selectedApiLevel {"17.06"} ;

    std::map<std::string, std::vector<ChangeSetHookFunction>> m_changesets ;
};

} // namespace simulation

} // namespace sofa

#endif
