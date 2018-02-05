/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <algorithm>
#include <sofa/version.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>

#include "SceneChecks.h"
#include "SceneCheckerVisitor.h"
#include "RequiredPlugin.h"


#include "APIVersion.h"
using sofa::component::APIVersion ;

#include "SceneCheckerVisitor.h"
#include "RequiredPlugin.h"

#include "APIVersion.h"
using sofa::component::APIVersion ;

namespace sofa
{
namespace simulation
{
using sofa::core::objectmodel::Base ;
using sofa::component::misc::RequiredPlugin ;
using sofa::core::ObjectFactory ;
using sofa::core::ExecParams ;
using sofa::helper::system::PluginRepository ;
using sofa::helper::system::PluginManager ;


SceneCheckerVisitor::SceneCheckerVisitor(const ExecParams* params) : Visitor(params)
{

}


SceneCheckerVisitor::~SceneCheckerVisitor()
{
}


void SceneCheckerVisitor::addCheck(SceneCheck::SPtr check)
{
    if( std::find(m_checkset.begin(), m_checkset.end(), check) == m_checkset.end() )
        m_checkset.push_back(check) ;
}


void SceneCheckerVisitor::removeCheck(SceneCheck::SPtr check)
{
    m_checkset.erase( std::remove( m_checkset.begin(), m_checkset.end(), check ), m_checkset.end() );
}

void SceneCheckerVisitor::validate(Node* node)
{
    std::stringstream tmp;
    for(SceneCheck::SPtr& check : m_checkset)
    {
        tmp << check->getName() << ", " ;
        check->doInit(node) ;
    }

    msg_info("SceneChecker") << "Validating node '"<< node->getPathName() << " with: [" << tmp.str() << "]" ;

    execute(node) ;

    for(SceneCheck::SPtr& check : m_checkset)
    {
        check->doPrintSummary() ;
    }
}


Visitor::Result SceneCheckerVisitor::processNodeTopDown(Node* node)
{
    for(SceneCheck::SPtr& check : m_checkset)
    {
        check->doCheckOn(node) ;
    }

    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

