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
#ifndef SOFA_SIMULATION_SCENECHECKS_H
#define SOFA_SIMULATION_SCENECHECKS_H

#include "config.h"

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

class SceneCheck
{
public:
    virtual const std::string getName() = 0 ;
    virtual const std::string getDesc() = 0 ;
    virtual void doCheckOn(Node* node) = 0 ;
};

class SceneCheckDuplicatedName : public SceneCheck
{
public:
    virtual const std::string getName() override ;
    virtual const std::string getDesc() override ;
    virtual void doCheckOn(Node* node) override ;
};

class SceneCheckMissingRequiredPlugin : public SceneCheck
{
public:
    virtual const std::string getName() override ;
    virtual const std::string getDesc() override ;
    virtual void doCheckOn(Node* node) override ;
};

class SceneCheckAPIChange : public SceneCheck
{
public:
    virtual const std::string getName() override ;
    virtual const std::string getDesc() override ;
    virtual void doCheckOn(Node* node) override ;
};

} /// _scenechecks_

using _scenechecks_::SceneCheck ;
using _scenechecks_::SceneCheckDuplicatedName ;
using _scenechecks_::SceneCheckMissingRequiredPlugin ;
using _scenechecks_::SceneCheckAPIChange ;

} /// namespace simulation

} /// namespace sofa

#endif /// SOFA_SIMULATION_SCENECHECKS_H
