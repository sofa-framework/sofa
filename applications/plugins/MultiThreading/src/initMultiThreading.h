/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef PLUGIN_MULTITHREADING_INIT_H
#define PLUGIN_MULTITHREADING_INIT_H

#include <sofa/SofaGeneral.h>

#ifndef WIN32
    #define SOFA_EXPORT_DYNAMIC_LIBRARY 
    #define SOFA_IMPORT_DYNAMIC_LIBRARY
    #define SOFA_MULTITHREADING_PLUGIN_API
#else
    #ifdef SOFA_MULTITHREADING_PLUGIN
		#define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
		#define SOFA_MULTITHREADING_PLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
    #else
		#define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
		#define SOFA_MULTITHREADING_PLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
    #endif
#endif


namespace sofa
{
namespace component
{
	extern "C" {
                SOFA_MULTITHREADING_PLUGIN_API void initExternalModule();
                SOFA_MULTITHREADING_PLUGIN_API const char* getModuleName();
                SOFA_MULTITHREADING_PLUGIN_API const char* getModuleVersion();
                SOFA_MULTITHREADING_PLUGIN_API const char* getModuleLicense();		
                SOFA_MULTITHREADING_PLUGIN_API const char* getModuleDescription();
                SOFA_MULTITHREADING_PLUGIN_API const char* getModuleComponentList();
	}

} //component
} //sofa 


#endif /* PLUGIN_XICATH_INIT_H */
