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
#include <sofa/component/topology/container/dynamic/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::topology::container::dynamic
{

extern void registerDynamicSparseGridGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerDynamicSparseGridTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerDynamicSparseGridTopologyModifier(sofa::core::ObjectFactory* factory);
extern void registerEdgeSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerEdgeSetTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerEdgeSetTopologyModifier(sofa::core::ObjectFactory* factory);
extern void registerHexahedronSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerHexahedronSetTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerHexahedronSetTopologyModifier(sofa::core::ObjectFactory* factory);
extern void registerMultilevelHexahedronSetTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerPointSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerPointSetTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerPointSetTopologyModifier(sofa::core::ObjectFactory* factory);
extern void registerQuadSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerQuadSetTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerQuadSetTopologyModifier(sofa::core::ObjectFactory* factory);
extern void registerTetrahedronSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerTetrahedronSetTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerTetrahedronSetTopologyModifier(sofa::core::ObjectFactory* factory);
extern void registerTriangleSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerTriangleSetTopologyContainer(sofa::core::ObjectFactory* factory);
extern void registerTriangleSetTopologyModifier(sofa::core::ObjectFactory* factory);

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
    registerDynamicSparseGridGeometryAlgorithms(factory);
    registerDynamicSparseGridTopologyContainer(factory);
    registerDynamicSparseGridTopologyModifier(factory);
    registerEdgeSetGeometryAlgorithms(factory);
    registerEdgeSetTopologyContainer(factory);
    registerEdgeSetTopologyModifier(factory);
    registerHexahedronSetGeometryAlgorithms(factory);
    registerHexahedronSetTopologyContainer(factory);
    registerHexahedronSetTopologyModifier(factory);
    registerMultilevelHexahedronSetTopologyContainer(factory);
    registerPointSetGeometryAlgorithms(factory);
    registerPointSetTopologyContainer(factory);
    registerPointSetTopologyModifier(factory);
    registerQuadSetGeometryAlgorithms(factory);
    registerQuadSetTopologyContainer(factory);
    registerQuadSetTopologyModifier(factory);
    registerTetrahedronSetGeometryAlgorithms(factory);
    registerTetrahedronSetTopologyContainer(factory);
    registerTetrahedronSetTopologyModifier(factory);
    registerTriangleSetGeometryAlgorithms(factory);
    registerTriangleSetTopologyContainer(factory);
    registerTriangleSetTopologyModifier(factory);
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

} // namespace sofa::component::topology::container::dynamic
