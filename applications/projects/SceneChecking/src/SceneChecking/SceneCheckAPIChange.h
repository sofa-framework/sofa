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

#include <SceneChecking/config.h>

#include <sofa/version.h>
#include <string>
#include <map>
#include <vector>
#include <functional>
#include <sofa/simulation/SceneCheck.h>

namespace sofa::simulation
{
    class Node;
} // namespace sofa::simulation

namespace sofa::core::objectmodel
{
    class Base;
} // namespace sofa::core::objectmodel

namespace sofa::_scenechecking_
{

typedef std::function<void(sofa::core::objectmodel::Base*)>     ChangeSetHookFunction;

class SOFA_SCENECHECKING_API SceneCheckAPIChange : public sofa::simulation::SceneCheck
{
public:
    SceneCheckAPIChange();
    virtual ~SceneCheckAPIChange();

    typedef std::shared_ptr<SceneCheckAPIChange> SPtr;
    static SPtr newSPtr() { return SPtr(new SceneCheckAPIChange()); }
    virtual const std::string getName() override;
    virtual const std::string getDesc() override;
    void doInit(sofa::simulation::Node* node) override;
    void doCheckOn(sofa::simulation::Node* node) override;
    void doPrintSummary() override;

    void installDefaultChangeSets();
    void addHookInChangeSet(const std::string& version, ChangeSetHookFunction fct);
private:
    std::string m_currentApiLevel;
    std::string m_selectedApiLevel {"17.06"};

    std::map<std::string, std::vector<ChangeSetHookFunction>> m_changesets;
};

} // namespace sofa::_scenechecking_

namespace sofa::scenechecking
{
    using _scenechecking_::SceneCheckAPIChange;
} // namespace sofa::component::scenechecking
