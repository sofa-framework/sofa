%module core

%include "std_string.i"

%{
#include "sofa/core/objectmodel/Base.h"
#include "sofa/core/objectmodel/BaseData.h"
#include "sofa/core/objectmodel/BaseObject.h"
#include "sofa/core/ObjectFactory.h"
%}

// TODO put this in a file to be included in sofa .i files
%typemap(out) sofa::core::objectmodel::BaseObject::SPtr {
    $result = SWIG_NewPointerObj($1.get(), SWIGTYPE_p_sofa__core__objectmodel__BaseObject, 0 );
}

namespace sofa
{
namespace core
{
namespace objectmodel
{

%nodefaultctor;
%nodefaultdtor;

class Base
{
public:
    const std::string& getName() const;
    void setName(const std::string& n);
    sofa::core::objectmodel::BaseData* findData( const std::string &name ) const;
};

class BaseData {
public:
    const std::string& getName() const;
    void setName(const std::string& name);
    virtual std::string getValueString() const = 0;
    virtual std::string getValueTypeString() const = 0;
};


class BaseObject : public virtual Base {
public:
    virtual std::string getPathName() const;
};

class BaseObjectDescription {
public:
    BaseObjectDescription(const char* name=NULL, const char* type=NULL);
    virtual ~BaseObjectDescription();

    virtual void setAttribute(const std::string& attr, const char* val);
};

class BaseContext;

} // objectmodel

class ObjectFactory {
public:
    static sofa::core::objectmodel::BaseObject::SPtr CreateObject(sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg);
};

} // core
} // sofa
