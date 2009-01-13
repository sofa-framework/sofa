#ifndef SOFA_CORE_H
#define SOFA_CORE_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_CORE
#	define SOFA_CORE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_CORE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
