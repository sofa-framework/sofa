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
#include <SofaCUDA/core/init.h>

#include <sofa/helper/system/PluginManager.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/gpu/cuda/mycuda.h>

namespace sofacuda::core
{

extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleLicense();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleDescription();
    SOFA_EXPORT_DYNAMIC_LIBRARY bool moduleIsInitialized();
    SOFA_EXPORT_DYNAMIC_LIBRARY void registerObjects(sofa::core::ObjectFactory* factory);
}

bool isModuleInitialized = false;

void init()
{
    static bool first = true;
    if (first)
    {
        isModuleInitialized = sofa::gpu::cuda::mycudaInit();
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
    return "A subset of SOFA componend using NVIDIA CUDA";
}

bool moduleIsInitialized()
{
    return isModuleInitialized;
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    SOFA_UNUSED(factory);
}

} // namespace sofacuda::core
