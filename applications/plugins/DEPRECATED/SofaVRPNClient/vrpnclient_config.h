/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * vrpnclient_config.h
 *
 *  Created on: 4 nov. 2009
 *      Author: froy
 */

#ifndef VRPNCLIENT_CONFIG_H_
#define VRPNCLIENT_CONFIG_H_

#include <sofa/helper/system/config.h>

#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SOFA_SOFAVRPNCLIENT_API
#else
#ifdef SOFA_BUILD_SOFAVRPNCLIENT
#define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#define SOFA_SOFAVRPNCLIENT_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#define SOFA_SOFAVRPNCLIENT_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif


#endif /* VRPNCLIENT_CONFIG_H_ */
