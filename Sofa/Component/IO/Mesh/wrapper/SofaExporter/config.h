#pragma once

#include <sofa/component/io/mesh/config.h>

#ifdef SOFA_BUILD_SOFAEXPORTER
#  define SOFA_SOFAEXPORTER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_SOFAEXPORTER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
