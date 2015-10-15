/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "ImageQtQuickGUI.h"

#include <QApplication>

#include "ImagePlaneView.h"
#include "ImagePlaneModel.h"

using namespace sofa::qtquick;

const int versionMajor = 1;
const int versionMinor = 0;

static void initResources()
{
    Q_INIT_RESOURCE(qml);
}

namespace sofa
{

namespace component
{

extern "C" {
    SOFA_IMAGE_QTQUICKGUI_API void initExternalModule();
    SOFA_IMAGE_QTQUICKGUI_API const char* getModuleName();
    SOFA_IMAGE_QTQUICKGUI_API const char* getModuleVersion();
    SOFA_IMAGE_QTQUICKGUI_API const char* getModuleLicense();
    SOFA_IMAGE_QTQUICKGUI_API const char* getModuleDescription();
    SOFA_IMAGE_QTQUICKGUI_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;

        initResources();

        qmlRegisterType<ImagePlaneModel>    ("ImagePlaneModel", versionMajor, versionMinor, "ImagePlaneModel");
        qmlRegisterType<ImagePlaneView>     ("ImagePlaneView",  versionMajor, versionMinor, "ImagePlaneView" );
    }
}

const char* getModuleName()
{
    return "Image Plugin - QtQuick GUI";
}

const char* getModuleVersion()
{
    return "0.1";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "Image QtQuick GUI";
}

const char* getModuleComponentList()
{
    return "";
}

} // namespace component

} // namespace sofa
