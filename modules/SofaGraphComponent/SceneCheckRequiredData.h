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
#ifndef SOFA_SIMULATION_SCENECHECKREQUIREDDATA_H
#define SOFA_SIMULATION_SCENECHECKREQUIREDDATA_H

#include "SceneCheck.h"

#include "config.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseData.h>

#include <map>
#include <vector>

namespace sofa
{
namespace simulation
{
namespace _scenechecking_
{
    
class SOFA_GRAPH_COMPONENT_API SceneCheckRequiredData : public SceneCheck
{
public:
    SceneCheckRequiredData();
    virtual ~SceneCheckRequiredData();

    typedef std::shared_ptr<SceneCheckRequiredData> SPtr;
    static SPtr newSPtr() { return SPtr(new SceneCheckRequiredData()); }
    virtual const std::string getName() override;
    virtual const std::string getDesc() override;
    virtual void doInit(Node* node) override;
    virtual void doCheckOn(Node* node) override;
    virtual void doPrintSummary() override;

private:
    std::map<sofa::core::objectmodel::BaseObject*, std::vector<sofa::core::objectmodel::BaseData*>> m_missingDatas;
};

} // namespace _scenechecking_

namespace scenechecking
{
    using _scenechecking_::SceneCheckRequiredData;
}

} // namespace simulation
} // namespace sofa

#endif // SOFA_SIMULATION_SCENECHECKREQUIREDDATA_H
