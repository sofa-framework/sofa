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
#include <sofa/component/io/mesh/init.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component::io::mesh
{

extern void registerBlenderExporter(sofa::core::ObjectFactory* factory);
extern void registerGIDMeshLoader(sofa::core::ObjectFactory* factory);
extern void registerGridMeshCreator(sofa::core::ObjectFactory* factory);
extern void registerMeshExporter(sofa::core::ObjectFactory* factory);
extern void registerMeshGmshLoader(sofa::core::ObjectFactory* factory);
extern void registerMeshOBJLoader(sofa::core::ObjectFactory* factory);
extern void registerMeshOffLoader(sofa::core::ObjectFactory* factory);
extern void registerMeshSTLLoader(sofa::core::ObjectFactory* factory);
extern void registerMeshTrianLoader(sofa::core::ObjectFactory* factory);
extern void registerMeshVTKLoader(sofa::core::ObjectFactory* factory);
extern void registerMeshXspLoader(sofa::core::ObjectFactory* factory);
extern void registerOffSequenceLoader(sofa::core::ObjectFactory* factory);
extern void registerSphereLoader(sofa::core::ObjectFactory* factory);
extern void registerSTLExporter(sofa::core::ObjectFactory* factory);
extern void registerStringMeshCreator(sofa::core::ObjectFactory* factory);
extern void registerVisualModelOBJExporter(sofa::core::ObjectFactory* factory);
extern void registerVoxelGridLoader(sofa::core::ObjectFactory* factory);
extern void registerVTKExporter(sofa::core::ObjectFactory* factory);

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
    registerBlenderExporter(factory);
    registerGIDMeshLoader(factory);
    registerGridMeshCreator(factory);
    registerMeshExporter(factory);
    registerMeshGmshLoader(factory);
    registerMeshOBJLoader(factory);
    registerMeshOffLoader(factory);
    registerMeshSTLLoader(factory);
    registerMeshTrianLoader(factory);
    registerMeshVTKLoader(factory);
    registerMeshXspLoader(factory);
    registerOffSequenceLoader(factory);
    registerSphereLoader(factory);
    registerSTLExporter(factory);
    registerStringMeshCreator(factory);
    registerVisualModelOBJExporter(factory);
    registerVoxelGridLoader(factory);
    registerVTKExporter(factory);
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

} // namespace sofa::component::io::mesh
