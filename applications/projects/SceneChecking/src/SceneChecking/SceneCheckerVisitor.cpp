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
#include "SceneCheckerVisitor.h"

#include <algorithm>
#include <sofa/simulation/Node.h>

namespace sofa::_scenechecking_
{
using sofa::core::ExecParams ;

SceneCheckerVisitor::SceneCheckerVisitor(const ExecParams* params) : Visitor(params)
{

}


SceneCheckerVisitor::~SceneCheckerVisitor()
{
}


void SceneCheckerVisitor::addCheck(sofa::simulation::SceneCheck::SPtr check)
{
    if( std::find(m_checkset.begin(), m_checkset.end(), check) == m_checkset.end() )
        m_checkset.push_back(check) ;
}


void SceneCheckerVisitor::removeCheck(sofa::simulation::SceneCheck::SPtr check)
{
    m_checkset.erase( std::remove( m_checkset.begin(), m_checkset.end(), check ), m_checkset.end() );
}

void SceneCheckerVisitor::validate(sofa::simulation::Node* node, simulation::SceneLoader* sceneLoader)
{
    std::stringstream tmp;
    bool first = true;
    for(const sofa::simulation::SceneCheck::SPtr& check : m_checkset)
    {
        tmp << (first ? "" : ", ") << check->getName() ;
        first = false;
    }
    msg_info("SceneCheckerVisitor") << "Validating node \""<< node->getName() << "\" with checks: [" << tmp.str() << "]" ;

    for(const sofa::simulation::SceneCheck::SPtr& check : m_checkset)
    {
        check->init(node, sceneLoader) ;
    }

    execute(node) ;

    for(const sofa::simulation::SceneCheck::SPtr& check : m_checkset)
    {
        check->printSummary(sceneLoader) ;
    }
    msg_info("SceneCheckerVisitor") << "Finished validating node \""<< node->getName() << "\".";
}


sofa::simulation::Visitor::Result SceneCheckerVisitor::processNodeTopDown(sofa::simulation::Node* node)
{
    for(const sofa::simulation::SceneCheck::SPtr& check : m_checkset)
    {
        check->doCheckOn(node) ;
    }

    return RESULT_CONTINUE;
}

} // namespace sofa::_scenechecking_
