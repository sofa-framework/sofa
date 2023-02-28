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
#include <sofa/simulation/SceneCheck.h>

#include <map>
#include <vector>

namespace sofa::_scenechecking_
{
    
class SOFA_SCENECHECKING_API SceneCheckUsingAlias : public sofa::simulation::SceneCheck
{
public:
    SceneCheckUsingAlias();
    virtual ~SceneCheckUsingAlias();

    typedef std::shared_ptr<SceneCheckUsingAlias> SPtr;
    static SPtr newSPtr() { return SPtr(new SceneCheckUsingAlias()); }
    virtual const std::string getName() override;
    virtual const std::string getDesc() override;
    void doInit(sofa::simulation::Node* node) override { SOFA_UNUSED(node); }
    void doCheckOn(sofa::simulation::Node* node) override { SOFA_UNUSED(node); }
    void doPrintSummary() override;

private:
    std::map<std::string, std::vector<std::string>> m_componentsCreatedUsingAlias;
};

} // namespace sofa::_scenechecking_

namespace sofa::scenechecking
{
    using _scenechecking_::SceneCheckUsingAlias;
} // namespace sofa::scenechecking

