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
#include <sofa/core/ExecParams.h>
#include <sofa/simulation/SceneLoaderFactory.h>

#include <functional>
#include <map>

#include <sofa/simulation/Visitor.h>

namespace sofa::_scenechecking_
{

class SOFA_SCENECHECKING_API SceneCheckerVisitor : public sofa::simulation::Visitor
{
public:
    SceneCheckerVisitor(const sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance()) ;
    ~SceneCheckerVisitor() override;

    void validate(sofa::simulation::Node* node, simulation::SceneLoader* sceneLoader) ;
    Result processNodeTopDown(sofa::simulation::Node* node) override ;

    void addCheck(sofa::simulation::SceneCheck::SPtr check) ;
    void removeCheck(sofa::simulation::SceneCheck::SPtr check) ;

private:
    std::vector<sofa::simulation::SceneCheck::SPtr> m_checkset ;
};

} // namespace sofa::_scenechecking_

namespace sofa::scenechecking
{
    using _scenechecking_::SceneCheckerVisitor;
} // namespace sofa::scenechecking
