#pragma once

#include <sofa/component/io/mesh/config.h>

#ifdef SOFA_BUILD_SOFAEXPORTER
#  define SOFA_SOFAEXPORTER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_SOFAEXPORTER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

// DEPRECATION MACROS

#define SOFA_ATTRIBUTE_DEPRECATED__SOFAEXPORTER_NAMESPACE_1712() \
    SOFA_ATTRIBUTE_DEPRECATED( \
        "v17.12 (PR#372)", "v21.12", \
        "This class is now in sofa::component::exporter namespace. ")
#define SOFA_ATTRIBUTE_DEPRECATED__SOFAEXPORTER_NAMESPACE_2106() \
    SOFA_ATTRIBUTE_DEPRECATED( \
        "v21.06", "v21.12", \
        "This class is now in sofa::component::exporter namespace. ")

#define SOFA_ATTRIBUTE_DISABLED__SOFAEXPORTER_NAMESPACE_1712() \
    SOFA_ATTRIBUTE_DISABLED( \
        "v17.12 (PR#372)", "v21.12", \
        "This class is now in sofa::component::exporter namespace. ")
#define SOFA_ATTRIBUTE_DISABLED__SOFAEXPORTER_NAMESPACE_2106() \
    SOFA_ATTRIBUTE_DISABLED( \
        "v21.06", "v21.12", \
        "This class is now in sofa::component::exporter namespace. ")
