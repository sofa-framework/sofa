#pragma once

#include <sofa/component/io/mesh/config.h>

#ifdef SOFA_BUILD_SOFALOADER
#  define SOFA_SOFALOADER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_SOFALOADER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
