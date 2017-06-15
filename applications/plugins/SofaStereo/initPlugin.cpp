/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "initPlugin.h"

#include <sofa/helper/system/FileRepository.h>

#ifndef STEREO_PLUGIN_DATA_DIR
#define STEREO_PLUGIN_DATA_DIR ""
#endif

namespace sofa
{

namespace component
{

	//Here are just several convenient functions to help user to know what contains the plugin

	extern "C" {
                SOFA_STEREO_API void initExternalModule();
                SOFA_STEREO_API const char* getModuleName();
                SOFA_STEREO_API const char* getModuleVersion();
                SOFA_STEREO_API const char* getModuleLicense();
                SOFA_STEREO_API const char* getModuleDescription();
                SOFA_STEREO_API const char* getModuleComponentList();
	}
	
	void initExternalModule()
	{
		static bool first = true;
        if (first)
        {
            first = false;

            sofa::helper::system::DataRepository.addLastPath(std::string(STEREO_PLUGIN_DATA_DIR));
            sofa::helper::system::DataRepository.addLastPath(std::string(STEREO_PLUGIN_DATA_DIR) + "/data");
        }
	}

	const char* getModuleName()
	{
                 return "Stereo plugin by Santi";
	}

	const char* getModuleVersion()
	{
                return "0.0";
	}

	const char* getModuleLicense()
	{
		return "LGPL";
	}


	const char* getModuleDescription()
	{
        return "Registration code";
	}

	const char* getModuleComponentList()
	{
                return "...";
	}



} 

} 


