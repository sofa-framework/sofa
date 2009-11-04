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
#ifdef SOFA_BUILD_PLUGINEXAMPLE
#define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#define SOFA_SOFAVRPNCLIENT_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#define SOFA_SOFAVRPNCLIENT_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif


#endif /* VRPNCLIENT_CONFIG_H_ */
