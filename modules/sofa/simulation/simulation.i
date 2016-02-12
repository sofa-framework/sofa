%module simulation

%include "std_string.i"

%{
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObject.h>
    
#include <sofa/simulation/common/Node.h>
%}


namespace sofa
{
namespace core 
{

%nodefaultctor ObjectFactory;
%nodefaultdtor ObjectFactory;
class ObjectFactory {
public:
    static sofa::core::objectmodel::BaseObject::SPtr CreateObject(sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg);
};
    
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

class BaseObjectDescription {
public:
    BaseObjectDescription(const char* name=NULL, const char* type=NULL);
    virtual void setAttribute(const std::string& attr, const char* val);
};

%nodefaultctor Base;
%nodefaultdtor Base;
class Base {
public:
    sofa::core::objectmodel::BaseData* findData( const std::string &name ) const;
};

%nodefaultctor BaseContext;
%nodefaultdtor BaseContext;
class BaseContext;

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
    const sofa::core::objectmodel::BaseContext* getContext() const;
};

}
}

%pythoncode %{
# mimic SofaPython Node.createObject method
def Node_createObject(self, type, **kwargs):
    desc = BaseObjectDescription(type, type)
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            desc.setAttribute(key, value)
    return ObjectFactory.CreateObject(self.getContext(), desc)
# turn the function into a Node method
Node.__dict__["createObject"]=Node_createObject
%}


