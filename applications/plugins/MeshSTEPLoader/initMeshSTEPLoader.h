#ifndef INITMeshSTEPLoader_H
#define INITMeshSTEPLoader_H


#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_MeshSTEPLoader
#define SOFA_MeshSTEPLoader_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_MeshSTEPLoader_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
