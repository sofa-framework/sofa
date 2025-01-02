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
#include <MultiThreading/config.h>
#include <MultiThreading/initMultiThreading.h>

#include <sofa/component/linearsolver/iterative/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>


namespace sofa::core
{
extern void registerDataExchange(sofa::core::ObjectFactory* factory);
}

namespace sofa::component::engine
{
extern void registerMeanComputationEngine(sofa::core::ObjectFactory* factory);
}
    
namespace multithreading
{

namespace component::animationloop
{
extern void registerAnimationLoopParallelScheduler(sofa::core::ObjectFactory* factory);
}

namespace component::collision::detection::algorithm
{
extern void registerParallelBVHNarrowPhase(sofa::core::ObjectFactory* factory);
extern void registerParallelBruteForceBroadPhase(sofa::core::ObjectFactory* factory);
}
    
namespace component::linearsolver::iterative
{
extern void registerParallelCGLinearSolver(sofa::core::ObjectFactory* factory);
}

namespace component::mapping::linear
{
extern void registerBeamLinearMapping_mt(sofa::core::ObjectFactory* factory);
}

namespace component::forcefield::solidmechanics::fem::elastic
{
extern void registerParallelHexahedronFEMForceField(sofa::core::ObjectFactory* factory);
extern void registerParallelTetrahedronFEMForceField(sofa::core::ObjectFactory* factory);
}

namespace component::solidmechanics::spring
{
extern void registerParallelMeshSpringForceField(sofa::core::ObjectFactory* factory);
extern void registerParallelSpringForceField(sofa::core::ObjectFactory* factory);
}

extern "C" {
SOFA_MULTITHREADING_PLUGIN_API void initExternalModule();
SOFA_MULTITHREADING_PLUGIN_API const char* getModuleName();
SOFA_MULTITHREADING_PLUGIN_API const char* getModuleVersion();
SOFA_MULTITHREADING_PLUGIN_API const char* getModuleLicense();
SOFA_MULTITHREADING_PLUGIN_API const char* getModuleDescription();
SOFA_MULTITHREADING_PLUGIN_API void registerObjects(sofa::core::ObjectFactory* factory);
}

void init()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);
        
        sofa::component::linearsolver::iterative::init();
        first = false;
    }
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "MultiThreading SOFA Framework";
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    sofa::core::registerDataExchange(factory);
    sofa::component::engine::registerMeanComputationEngine(factory);
    multithreading::component::animationloop::registerAnimationLoopParallelScheduler(factory);
    multithreading::component::collision::detection::algorithm::registerParallelBVHNarrowPhase(factory);
    multithreading::component::collision::detection::algorithm::registerParallelBruteForceBroadPhase(factory);
    multithreading::component::linearsolver::iterative::registerParallelCGLinearSolver(factory);
    multithreading::component::mapping::linear::registerBeamLinearMapping_mt(factory);
    multithreading::component::forcefield::solidmechanics::fem::elastic::registerParallelHexahedronFEMForceField(factory);
    multithreading::component::forcefield::solidmechanics::fem::elastic::registerParallelTetrahedronFEMForceField(factory);
    multithreading::component::solidmechanics::spring::registerParallelMeshSpringForceField(factory);
    multithreading::component::solidmechanics::spring::registerParallelSpringForceField(factory);
    
}

}
