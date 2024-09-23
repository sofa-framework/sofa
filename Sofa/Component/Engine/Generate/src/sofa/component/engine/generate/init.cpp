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
#include <sofa/component/engine/generate/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::engine::generate
{

extern void registerExtrudeEdgesAndGenerateQuads(sofa::core::ObjectFactory* factory);
extern void registerExtrudeQuadsAndGenerateHexas(sofa::core::ObjectFactory* factory);
extern void registerExtrudeSurface(sofa::core::ObjectFactory* factory);
extern void registerGenerateCylinder(sofa::core::ObjectFactory* factory);
extern void registerGenerateGrid(sofa::core::ObjectFactory* factory);
extern void registerGenerateRigidMass(sofa::core::ObjectFactory* factory);
extern void registerGenerateSphere(sofa::core::ObjectFactory* factory);
extern void registerGroupFilterYoungModulus(sofa::core::ObjectFactory* factory);
extern void registerJoinPoints(sofa::core::ObjectFactory* factory);
extern void registerMergeMeshes(sofa::core::ObjectFactory* factory);
extern void registerMergePoints(sofa::core::ObjectFactory* factory);
extern void registerMergeSets(sofa::core::ObjectFactory* factory);
extern void registerMergeVectors(sofa::core::ObjectFactory* factory);
extern void registerMeshBarycentricMapperEngine(sofa::core::ObjectFactory* factory);
extern void registerMeshClosingEngine(sofa::core::ObjectFactory* factory);
extern void registerMeshTetraStuffing(sofa::core::ObjectFactory* factory);
extern void registerNormalsFromPoints(sofa::core::ObjectFactory* factory);
extern void registerNormEngine(sofa::core::ObjectFactory* factory);
extern void registerRandomPointDistributionInSurface(sofa::core::ObjectFactory* factory);
extern void registerSpiral(sofa::core::ObjectFactory* factory);

extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_EXPORT_DYNAMIC_LIBRARY void registerObjects(sofa::core::ObjectFactory* factory);
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

void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerExtrudeEdgesAndGenerateQuads(factory);
    registerExtrudeQuadsAndGenerateHexas(factory);
    registerExtrudeSurface(factory);
    registerGenerateCylinder(factory);
    registerGenerateGrid(factory);
    registerGenerateRigidMass(factory);
    registerGenerateSphere(factory);
    registerGroupFilterYoungModulus(factory);
    registerJoinPoints(factory);
    registerMergeMeshes(factory);
    registerMergePoints(factory);
    registerMergeSets(factory);
    registerMergeVectors(factory);
    registerMeshBarycentricMapperEngine(factory);
    registerMeshClosingEngine(factory);
    registerMeshTetraStuffing(factory);
    registerNormalsFromPoints(factory);
    registerNormEngine(factory);
    registerRandomPointDistributionInSurface(factory);
    registerSpiral(factory);
}

void init()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        first = false;
    }
}

} // namespace sofa::component::engine::generate
