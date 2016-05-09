#ifndef PLUGINA_H
#define PLUGINA_H

#include <sofa/helper/system/config.h>

#include <sofa/core/objectmodel/BaseObject.h>

class Foo: public sofa::core::objectmodel::BaseObject {

};

template <class C>
class Bar: public sofa::core::objectmodel::BaseObject {

};

#ifdef SOFA_BUILD_PLUGINA
#define SOFA_PluginA_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_PluginA_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
