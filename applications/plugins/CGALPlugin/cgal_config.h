/*
 * cgal_config.h
 *
 *  Created on: 27 oct. 2009
 *      Author: froy
 */

#ifndef CGAL_CONFIG_H_
#define CGAL_CONFIG_H_

#include <sofa/helper/system/config.h>

#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SOFA_CGALPLUGIN_API
#else
#ifdef SOFA_BUILD_CGALPLUGIN
#define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#define SOFA_CGALPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#define SOFA_CGALPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif

#endif /* CGAL_CONFIG_H_ */
