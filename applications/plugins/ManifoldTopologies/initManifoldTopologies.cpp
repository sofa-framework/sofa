/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <ManifoldTopologies/ManifoldTopologies.h>
#include "ManifoldTopologyObject_double.h"
#include "ManifoldTopologyObject_float.h"

#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SOFA_PLUGIN_API
#else
#ifdef SOFA_BUILD_MANIFOLDTOPOLOGIES
#define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#define SOFA_PLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#define SOFA_PLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_PLUGIN_API void initExternalModule();
    SOFA_PLUGIN_API const char* getModuleName();
    SOFA_PLUGIN_API const char* getModuleVersion();
    SOFA_PLUGIN_API const char* getModuleLicense();
    SOFA_PLUGIN_API const char* getModuleDescription();
    SOFA_PLUGIN_API const char* getModuleComponentList();
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
    return "Manifold Topologies Plugin";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "?";
}

const char* getModuleDescription()
{
    return "Use Manifold Topologies functionnalities into SOFA";
}

const char* getModuleComponentList()
{
    return "ManifoldEdge, ManifoldTriangle, ManifoldTetrahedron";
}



}

}

SOFA_LINK_CLASS(ManifoldEdgeSetGeometryAlgorithms)
SOFA_LINK_CLASS(ManifoldEdgeSetTopologyAlgorithms)
SOFA_LINK_CLASS(ManifoldEdgeSetTopologyContainer)
SOFA_LINK_CLASS(ManifoldEdgeSetTopologyModifier)
SOFA_LINK_CLASS(ManifoldTriangleSetTopologyContainer)
SOFA_LINK_CLASS(ManifoldTriangleSetTopologyModifier)
SOFA_LINK_CLASS(ManifoldTriangleSetTopologyAlgorithms)
//SOFA_LINK_CLASS(ManifoldTetrahedronSetTopologyContainer)
