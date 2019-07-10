/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/config.h>
#include <iostream>
#include <string>
#include <map>
#include <memory>

namespace sofa {
namespace simulation {
    class Node;
}
}

namespace sofa
{
namespace simulation
{
namespace _scenechecking_
{

class SOFA_GRAPH_COMPONENT_API SceneCheck
{
public:
    virtual ~SceneCheck() {}

    typedef std::shared_ptr<SceneCheck> SPtr;
    virtual const std::string getName() = 0;
    virtual const std::string getDesc() = 0;
    virtual void doInit(Node* node) { SOFA_UNUSED(node); }
    virtual void doCheckOn(Node* node) = 0;
    virtual void doPrintSummary() {}
};

} // namespace _scenechecking_

namespace scenechecking
{
    using _scenechecking_::SceneCheck;
}

} // namespace simulation
} // namespace sofa

#endif // SOFA_SIMULATION_SCENECHECKS_H
