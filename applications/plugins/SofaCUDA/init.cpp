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
#include <SofaCUDA/config.h>
#include <SofaCUDA/init.h>
#include <sofa/gpu/cuda/mycuda.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gpu::cuda
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
SOFA_GPU_CUDA_API void initExternalModule();
SOFA_GPU_CUDA_API const char* getModuleName();
SOFA_GPU_CUDA_API const char* getModuleVersion();
SOFA_GPU_CUDA_API const char* getModuleLicense();
SOFA_GPU_CUDA_API const char* getModuleDescription();
SOFA_GPU_CUDA_API const char* getModuleComponentList();
SOFA_GPU_CUDA_API bool moduleIsInitialized();
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
    return "SofaCUDA";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "GPU-based computing using NVIDIA CUDA";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = sofa::core::ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

bool moduleIsInitialized()
{
    return isModuleInitialized;
}

}
