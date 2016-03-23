/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <CGALPlugin/config.h>

#include <MeshGenerationFromPolyhedron.h>

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help users know what the plugin contains 

extern "C" {
    SOFA_CGALPLUGIN_API void initExternalModule();
    SOFA_CGALPLUGIN_API const char* getModuleName();
    SOFA_CGALPLUGIN_API const char* getModuleVersion();
    SOFA_CGALPLUGIN_API const char* getModuleLicense();
    SOFA_CGALPLUGIN_API const char* getModuleDescription();
    SOFA_CGALPLUGIN_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return "CGAL Plugin";
}

const char* getModuleVersion()
{
    return "0.2";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "Use CGAL functionnalities into SOFA";
}

const char* getModuleComponentList()
{
    return "MeshGenerationFromPolyhedron, TriangularConvexHull3D";
}



}

}


SOFA_LINK_CLASS(MeshGenerationFromPolyhedron)
SOFA_LINK_CLASS(TriangularConvexHull3D)

