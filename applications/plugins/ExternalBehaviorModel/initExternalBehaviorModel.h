#ifndef INITExternalBehaviorModel_H
#define INITExternalBehaviorModel_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_ExternalBehaviorModel
#define SOFA_ExternalBehaviorModel_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_ExternalBehaviorModel_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif // INITExternalBehaviorModel_H
