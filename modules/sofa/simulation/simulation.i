%module simulation

%include "std_string.i"

%{
#include <sofa/simulation/common/Node.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObject.h>
%}

namespace sofa
{
namespace core 
{
namespace objectmodel
{

%typemap(out) sofa::core::objectmodel::BaseObject::SPtr {
    $result = SWIG_NewPointerObj($1.get(), SWIGTYPE_p_sofa__core__objectmodel__BaseObject, SWIG_POINTER_DISOWN |  0 );
}

%nodefaultctor BaseData;
%nodefaultdtor BaseData;
class BaseData {
public:
    virtual std::string getValueString() const = 0;
    virtual std::string getValueTypeString() const = 0;
};

%nodefaultctor Base;
%nodefaultdtor Base;
class Base {
public:
    sofa::core::objectmodel::BaseData* findData( const std::string &name ) const;
};

%nodefaultctor BaseObject;
%nodefaultdtor BaseObject;
class BaseObject : public virtual Base {
public:
    virtual std::string getPathName() const;
};

} // objectmodel
} // core

namespace simulation
{

%typemap(out) sofa::simulation::Node::SPtr {
    $result = SWIG_NewPointerObj($1.get(), SWIGTYPE_p_sofa__simulation__Node, SWIG_POINTER_DISOWN |  0 );
}

%nodefaultctor Node;
%nodefaultdtor Node;
class Node {
public:
    virtual sofa::simulation::Node::SPtr createChild(const std::string& nodeName)=0;
};

%extend Node {
    sofa::core::objectmodel::BaseObject::SPtr createObject(std::string const& type) {
        sofa::core::objectmodel::BaseContext* context = self->toBaseContext();
        sofa::core::objectmodel::BaseObjectDescription desc(type.c_str(),type.c_str());
        sofa::core::objectmodel::BaseObject::SPtr obj = sofa::core::ObjectFactory::getInstance()->createObject(context,&desc);
        if (obj==0) {
        }
        return obj.get();
    }
}

}
}


