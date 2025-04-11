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
#include <SofaDistanceGrid/CUDA/init.h>
#include <SofaDistanceGrid/initSofaDistanceGrid.h>
#include <SofaCUDA/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::gpu::cuda
{
    extern void registerCudaCollisionDetection(sofa::core::ObjectFactory* factory);
    extern void registerCudaRigidDistanceGridCollisionModel(sofa::core::ObjectFactory* factory);
}

namespace sofadistancegrid::cuda
{

extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_SOFADISTANCEGRID_API void registerObjects(sofa::core::ObjectFactory* factory);
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

void init()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        sofadistancegrid::initSofaDistanceGrid();
        sofa::gpu::cuda::init();
        first = false;
    }
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    sofa::gpu::cuda::registerCudaCollisionDetection(factory);
    sofa::gpu::cuda::registerCudaRigidDistanceGridCollisionModel(factory);
}

} // namespace volumetricrendering::cuda
