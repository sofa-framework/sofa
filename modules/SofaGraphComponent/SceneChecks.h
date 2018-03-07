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
#ifndef SOFA_SIMULATION_SCENECHECKS_H
#define SOFA_SIMULATION_SCENECHECKS_H

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
namespace _scenechecks_
{

class SOFA_GRAPH_COMPONENT_API SceneCheck
{
public:
    typedef std::shared_ptr<SceneCheck> SPtr ;
    virtual const std::string getName() = 0 ;
    virtual const std::string getDesc() = 0 ;
    virtual void doInit(Node* node) { SOFA_UNUSED(node); }
    virtual void doCheckOn(Node* node) = 0 ;
    virtual void doPrintSummary() {}
};

class SOFA_GRAPH_COMPONENT_API SceneCheckDuplicatedName : public SceneCheck
{
public:
    typedef std::shared_ptr<SceneCheckDuplicatedName> SPtr ;
    static SPtr newSPtr() { return SPtr(new SceneCheckDuplicatedName()); }
    virtual const std::string getName() override ;
    virtual const std::string getDesc() override ;
    virtual void doInit(Node* node) override ;
    virtual void doCheckOn(Node* node) override ;
    virtual void doPrintSummary() override ;

private:
    bool m_hasDuplicates ;
    std::stringstream m_duplicatedMsg ;
};

class SOFA_GRAPH_COMPONENT_API SceneCheckMissingRequiredPlugin : public SceneCheck
{
public:
    typedef std::shared_ptr<SceneCheckMissingRequiredPlugin > SPtr ;
    static SPtr newSPtr() { return SPtr(new SceneCheckMissingRequiredPlugin()); }
    virtual const std::string getName() override ;
    virtual const std::string getDesc() override ;
    virtual void doInit(Node* node) override ;
    virtual void doCheckOn(Node* node) override ;
    virtual void doPrintSummary() override ;

private:    
    std::map<std::string, bool > m_loadedPlugins ;
    std::map<std::string, std::vector<std::string> > m_requiredPlugins ;
};


} /// _scenechecks_

using _scenechecks_::SceneCheck ;
using _scenechecks_::SceneCheckDuplicatedName ;
using _scenechecks_::SceneCheckMissingRequiredPlugin ;

namespace scenecheckers
{
    using _scenechecks_::SceneCheck ;
    using _scenechecks_::SceneCheckDuplicatedName ;
    using _scenechecks_::SceneCheckMissingRequiredPlugin ;
} /// checkers

} /// namespace simulation

} /// namespace sofa

#endif /// SOFA_SIMULATION_SCENECHECKS_H
