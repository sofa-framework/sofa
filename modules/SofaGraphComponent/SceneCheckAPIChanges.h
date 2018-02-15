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
#ifndef SOFA_SIMULATION_SCENECHECKAPICHANGES_H
#define SOFA_SIMULATION_SCENECHECKAPICHANGES_H

#include "SceneChecks.h"
#include "config.h"
#include <map>

/////////////////////////////// FORWARD DECLARATION ////////////////////////////////////////////////
namespace sofa {
    namespace simulation {
        class Node ;
    }
}


/////////////////////////////////////// DECLARATION ////////////////////////////////////////////////
namespace sofa
{
namespace simulation
{
namespace _scenecheckapichange_
{


typedef std::function<void(sofa::core::objectmodel::Base*)> ChangeSetHookFunction ;
class SOFA_GRAPH_COMPONENT_API SceneCheckAPIChange : public SceneCheck
{
public:
    SceneCheckAPIChange() ;
    virtual ~SceneCheckAPIChange() ;

    typedef std::shared_ptr<SceneCheckAPIChange> SPtr ;
    static SPtr newSPtr() { return SPtr(new SceneCheckAPIChange()); }
    virtual const std::string getName() override ;
    virtual const std::string getDesc() override ;
    virtual void doInit(Node* node) override ;
    virtual void doCheckOn(Node* node) override ;

    void installDefaultChangeSets() ;
    void addHookInChangeSet(const std::string& version, ChangeSetHookFunction fct) ;
private:
    std::string m_currentApiLevel;
    std::string m_selectedApiLevel {"17.06"} ;

    std::map<std::string, std::vector<ChangeSetHookFunction>> m_changesets ;
};

} /// _scenechecks_

using _scenecheckapichange_::SceneCheckAPIChange ;

namespace scenecheckers
{
    using _scenecheckapichange_::SceneCheckAPIChange ;
} /// checkers

} /// namespace simulation

} /// namespace sofa

#endif /// SOFA_SIMULATION_SCENECHECKS_H
