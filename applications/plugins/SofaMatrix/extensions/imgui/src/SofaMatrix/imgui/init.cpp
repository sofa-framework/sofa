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
#include <SofaMatrix/imgui/init.h>
#include <sofa/core/ObjectFactory.h>

namespace sofamatriximgui
{

void initializePlugin() 
{
    static bool first = true;
    if (first) {
        first = false;
        // Register components here
    }
}

}

extern "C" 
{
    SOFAMATRIX_IMGUI_API void initExternalModule() 
    {
        sofamatriximgui::initializePlugin();
    }

    SOFAMATRIX_IMGUI_API const char* getModuleName() 
    {
        return sofamatriximgui::MODULE_NAME;
    }

    SOFAMATRIX_IMGUI_API const char* getModuleVersion() 
    {
        return sofamatriximgui::MODULE_VERSION;
    }

    SOFAMATRIX_IMGUI_API const char* getModuleLicense() 
    {
        return "LGPL";
    }

    SOFAMATRIX_IMGUI_API const char* getModuleDescription() 
    {
        return "SOFA plugin for SofaMatrix.imgui";
    }
}
