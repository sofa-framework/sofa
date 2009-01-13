#ifndef SOFA_HELPER_H
#define SOFA_HELPER_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_HELPER
#	define SOFA_HELPER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_HELPER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
