#ifndef PLUGINF_H
#define PLUGINF_H

#include <sofa/helper/system/config.h>

#include <sofa/core/objectmodel/BaseObject.h>

class FooF: public sofa::core::objectmodel::BaseObject {

};

template <class C>
class BarF: public sofa::core::objectmodel::BaseObject {

};

#ifdef SOFA_BUILD_PLUGINF
#define SOFA_Pluginf_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_Pluginf_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
