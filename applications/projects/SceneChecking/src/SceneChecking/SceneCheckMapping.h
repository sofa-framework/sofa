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
#include <sofa/type/vector.h>

namespace sofa::scenechecking
{

/**
 * This SceneCheck is motivated by the visitor mechanism that assumes:
 * If both a mechanical state and a mapping are present in a Node, the state is considered mapped,
 * no matter if they are related or not.
 */
class SOFA_SCENECHECKING_API SceneCheckMapping : public sofa::simulation::SceneCheck
{
public:
    ~SceneCheckMapping() override;
    typedef std::shared_ptr<SceneCheckMapping> SPtr;
    static SPtr newSPtr() { return std::make_shared<SceneCheckMapping>(); }
    const std::string getName() override;
    const std::string getDesc() override;
    void doInit(sofa::simulation::Node* node) override;
    void doCheckOn(sofa::simulation::Node* node) override;
    void doPrintSummary() override;

private:
    sofa::type::vector<sofa::simulation::Node*> m_nodesWithMappingNoState;
    sofa::type::vector<sofa::simulation::Node*> m_nodesWithMappingWrongState;
    sofa::type::vector<sofa::simulation::Node*> m_nodesWithMappingAndTwoStates;
};

}
