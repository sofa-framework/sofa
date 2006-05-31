#ifndef SOFA_COMPONENTS_COMMON_CONFIG_H
#define SOFA_COMPONENTS_COMMON_CONFIG_H

#ifdef WIN32
#define NOMINMAX
#include <windows.h>
/*
namespace std
{
template<class T> T min(const T& a, const T& b) { return _cpp_min(a,b); }
template<class T> T max(const T& a, const T& b) { return _cpp_max(a,b); }
}
*/

#define snprintf _snprintf
#endif


#define sofa_concat(a,b) a##b

#define SOFA_DECL_CLASS(name) extern "C" { int sofa_concat(class_,name) = 0; }
#define SOFA_LINK_CLASS(name) extern "C" { extern int sofa_concat(class_,name); int sofa_concat(link_,name) = sofa_concat(class_,name); }

#endif
