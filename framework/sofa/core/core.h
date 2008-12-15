#ifndef SOFA_CORE_H
#define SOFA_CORE_H

#ifndef WIN32
#	define SOFA_EXPORT_DYNAMIC_LIBRARY
#   define SOFA_IMPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#   define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#endif

#ifdef SOFA_BUILD_CORE
#	define SOFA_CORE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_CORE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
