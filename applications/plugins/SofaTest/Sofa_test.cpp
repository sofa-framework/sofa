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

#include "Sofa_test.h"
#include <SceneCreator/SceneCreator.h>

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::PluginRepository ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::PluginRepository;
using sofa::helper::system::FileSystem;

#include <sofa/helper/Utils.h>
using sofa::helper::Utils;

namespace sofa {
namespace {
    struct raii {
      raii() {
          PluginManager::getInstance().loadPlugin("SceneCreator") ;
          PluginManager::getInstance().loadPlugin("SofaComponentAll") ;
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



template class SOFA_SOFATEST_API Sofa_test<double>;

}
