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
#ifndef SOFA_SIMULATION_SCENECHECKDUPLICATEDNAME_H
#define SOFA_SIMULATION_SCENECHECKDUPLICATEDNAME_H

#include "config.h"
#include "SceneChecks.h"
#include <map>
#include <sstream>

namespace sofa
{
namespace simulation
{
namespace _scenechecking_
{
    
class SOFA_GRAPH_COMPONENT_API SceneCheckDuplicatedName : public SceneCheck
{
public:
    typedef std::shared_ptr<SceneCheckDuplicatedName> SPtr;
    static SPtr newSPtr() { return SPtr(new SceneCheckDuplicatedName()); }
    virtual const std::string getName() override;
    virtual const std::string getDesc() override;
    virtual void doInit(Node* node) override;
    virtual void doCheckOn(Node* node) override;
    virtual void doPrintSummary() override;

private:
    bool m_hasDuplicates;
    std::stringstream m_duplicatedMsg;
};

} // namespace _scenechecking_

namespace scenechecking
{
    using _scenechecking_::SceneCheckDuplicatedName;
}

} // namespace simulation
} // namespace sofa

#endif // SOFA_SIMULATION_SCENECHECKDUPLICATEDNAME_H
