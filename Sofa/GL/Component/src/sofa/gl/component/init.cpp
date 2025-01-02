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
#include <sofa/gl/component/init.h>

#include <sofa/gl/component/engine/init.h>
#include <sofa/gl/component/rendering2d/init.h>
#include <sofa/gl/component/rendering3d/init.h>
#include <sofa/gl/component/shader/init.h>

#include <sofa/helper/system/PluginManager.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gl::component
{
    
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
    factory->registerObjectsFromPlugin("Sofa.GL.Component.Engine");
    factory->registerObjectsFromPlugin("Sofa.GL.Component.Rendering2D");
    factory->registerObjectsFromPlugin("Sofa.GL.Component.Rendering3D");
    factory->registerObjectsFromPlugin("Sofa.GL.Component.Shader");
}

void init()
{
    static bool first = true;
    if (first)
    {
        // force dependencies at compile-time
        sofa::gl::component::engine::init();
        sofa::gl::component::rendering2d::init();
        sofa::gl::component::rendering3d::init();
        sofa::gl::component::shader::init();

        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(MODULE_NAME);

        first = false;
    }
}

} // namespace sofa::gl::component
