/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "Sofa_test.h"
#include <SceneCreator/SceneCreator.h>

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::PluginRepository ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::PluginRepository;
using sofa::helper::system::DataRepository;
using sofa::helper::system::FileSystem;

#include <sofa/helper/Utils.h>
using sofa::helper::Utils;

namespace sofa {
namespace {
    static struct raii {
      raii() {
          const std::string pluginDir = Utils::getPluginDirectory() ;
          PluginRepository.addFirstPath(pluginDir);
          PluginManager::getInstance().loadPlugin("SceneCreator") ;
          PluginManager::getInstance().loadPlugin("SofaAllCommonComponents") ;
      }
    } singleton;
}

BaseSofa_test::BaseSofa_test()
{
    dmsg_deprecated("Sofa_test") << "Sofa_test & BaseSofa_test are now deprecated classes. "
                                    "To fix this message you should replace their usage by BaseTest, NumericTest or BaseSimulationTest to implement your tests" ;
    modeling::initSofa();
}

BaseSofa_test::~BaseSofa_test()
{
    clearSceneGraph();
}

void BaseSofa_test::clearSceneGraph()
{
    modeling::clearScene();
}



#ifdef SOFA_WITH_FLOAT
template struct SOFA_SOFATEST_API Sofa_test<float>;
#endif
#ifdef SOFA_WITH_DOUBLE
template struct SOFA_SOFATEST_API Sofa_test<double>;
#endif
}
