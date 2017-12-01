#ifndef STENTEXP_H
#define STENTEXP_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_STENTEXP
#define SOFA_StentExp_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_StentExp_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

/**
TODO: Main page of the doxygen documentation for the plugin
*/

#endif
