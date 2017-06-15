/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/)
{
    BackTrace::autodump();

    std::cout << "Before A" << std::endl ;

    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

    std::cout << "Before" << std::endl ;
    std::vector<ObjectFactory::ClassEntry::SPtr> result;

    //ObjectFactory::getInstance()->dump() ;
    /*
    ObjectFactory::getInstance()->getAllEntries(result);

    std::cout << "End" << std::endl ;

    for(ObjectFactory::ClassEntry::SPtr& entry : result)
    {
        if(entry){
            std::cout << "Processing: " << entry->m_componentName << std::endl ;
            for(auto& aliasname : entry->aliases){
                std::cout << aliasname << " " << entry->m_componentName << std::endl ;
            }
        }
    }
*/
    return 0;
}
