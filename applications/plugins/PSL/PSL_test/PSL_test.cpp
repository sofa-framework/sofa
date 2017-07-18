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
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

/******************************************************************************
 * Contributors:                                                              *
 *    - damien.marchal@univ-lille1.fr                                         *
 *****************************************************************************/
#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

#include <SofaTest/TestMessageHandler.h>

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::simulation::graph::DAGSimulation;

#include <SofaGeneralEngine/SphereROI.h>
using sofa::component::engine::SphereROI ;

#include <sofa/core/visual/VisualParams.h>
using sofa::core::visual::VisualParams;

#include "../SceneLoaderPSL.h"
using sofa::simulation::SceneLoaderPSL ;
using sofa::core::objectmodel::BaseObject ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

using std::vector;
using std::string;

namespace
{

class PSL_test : public Sofa_test<>
{
public:
    void SetUp(){
        PluginManager::getInstance().loadPlugin("PSL") ;
        PluginManager::getInstance().loadPlugin("SofaPython") ;

        if( !sofa::simulation::getSimulation() )
            sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

    }

    void checkTestFiles(const std::string& filename)
    {

        static const std::string scenePath = std::string(PSL_TESTFILES_DIR)+filename;

        Node::SPtr root = sofa::simulation::getSimulation()->load(scenePath.c_str());

        sofa::simulation::getSimulation()->unload( root ) ;
    }

};

TEST_F(PSL_test, checkTestFiles)
{
    checkTestFiles("test_pythonlocals.psl") ;
}


}
