/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/testing/BaseSimulationTest.h>

#include <sofa/core/fwd.h>
using sofa::core::execparams::defaultInstance;

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory ;
using sofa::simulation::SceneLoader ;

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

namespace sofa::testing
{

bool BaseSimulationTest::importPlugin(const std::string& name)
{
    const auto status = PluginManager::getInstance().loadPlugin(name);
    return status == PluginManager::PluginLoadStatus::SUCCESS || status == PluginManager::PluginLoadStatus::ALREADY_LOADED;
}

BaseSimulationTest::SceneInstance::SceneInstance(const std::string& type, const std::string& desc)
{
    if(type != "xml"){
        msg_error("BaseSimulationTest") << "Unsupported scene of type '"<< type << "' currently only 'xml' type is supported." ;
        return ;
    }

    root = SceneLoaderXML::loadFromMemory("dynamicscene", desc.c_str()) ;
}

BaseSimulationTest::SceneInstance BaseSimulationTest::SceneInstance::LoadFromFile(const std::string& filename)
{
    BaseSimulationTest::SceneInstance instance ;
    assert(sofa::simulation::getSimulation());

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
    root = simulation::getSimulation()->createNewNode(rootname) ;
}

void BaseSimulationTest::SceneInstance::loadSceneFile(const std::string& filename)
{
    root = sofa::simulation::node::load(filename);
    
    if (root == nullptr)
        msg_error("BaseSimulationTest") << "Unable to find a valid loader for: '" << filename << "'";
}


BaseSimulationTest::SceneInstance::~SceneInstance()
{
    sofa::simulation::node::unload(root) ;
}

void BaseSimulationTest::SceneInstance::initScene()
{
    sofa::simulation::node::initRoot(root.get());
}

void BaseSimulationTest::SceneInstance::simulate(const double timestep)
{
    sofa::simulation::node::animate(root.get(), static_cast<SReal>(timestep) );
}

BaseSimulationTest::BaseSimulationTest()
{
    assert(sofa::simulation::getSimulation());
}

} // namespace sofa::testing
