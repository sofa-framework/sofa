/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
/******************************************************************************
 * Contributors:
 *    - damien.marchal@univ-lille1.fr
 ******************************************************************************/
#include "BaseSimulationTest.h"

#include <sofa/core/ExecParams.h>
using sofa::core::ExecParams ;

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory ;
using sofa::simulation::SceneLoader ;

#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

namespace sofa
{
namespace helper
{
namespace testing
{

bool BaseSimulationTest::importPlugin(const std::string& name)
{
    return PluginManager::getInstance().loadPlugin(name) ;
}

BaseSimulationTest::SceneInstance::SceneInstance(const std::string& type, const std::string& desc)
{
    if(type != "xml"){
        msg_error("BaseSimulationTest") << "Unsupported scene of type '"<< type << "' currently only 'xml' type is supported." ;
        return ;
    }

    if(simulation::getSimulation() == nullptr)
        simulation::setSimulation(new simulation::graph::DAGSimulation()) ;

    simulation = simulation::getSimulation() ;
    root = SceneLoaderXML::loadFromMemory("dynamicscene", desc.c_str(), desc.size()) ;
}

BaseSimulationTest::SceneInstance BaseSimulationTest::SceneInstance::LoadFromFile(const std::string& filename)
{
    BaseSimulationTest::SceneInstance instance ;
    if(simulation::getSimulation() == nullptr)
        simulation::setSimulation(new simulation::graph::DAGSimulation()) ;

    for(SceneLoader* loader : (*SceneLoaderFactory::getInstance()->getEntries()) )
    {
        if(loader->canLoadFileName(filename.c_str()))
        {
            instance.root = loader->load(filename.c_str()) ;
            return instance ;
        }
    }
    msg_error("BaseSimulationTest") << "Unable to find a valid loader for: '"<< filename << "'" ;
    return instance ;
}

BaseSimulationTest::SceneInstance::SceneInstance(const std::string& rootname)
{
    if(simulation::getSimulation() == nullptr)
        simulation::setSimulation(new simulation::graph::DAGSimulation()) ;

    simulation = simulation::getSimulation() ;
    root = simulation::getSimulation()->createNewNode(rootname) ;
}


BaseSimulationTest::SceneInstance::~SceneInstance()
{
    simulation::getSimulation()->unload(root) ;
}

void BaseSimulationTest::SceneInstance::initScene()
{
    root->init(ExecParams::defaultInstance()) ;
}

void BaseSimulationTest::SceneInstance::simulate(const double timestep)
{
    simulation->animate( root.get(), (SReal)timestep );
}

BaseSimulationTest::BaseSimulationTest()
{
    simulation::setSimulation(new simulation::graph::DAGSimulation()) ;
}

} ///testing
} ///helper
} ///sofa
