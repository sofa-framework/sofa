#ifndef SOFA_ABSTRACT_BASEOBJECT_H
#define SOFA_ABSTRACT_BASEOBJECT_H

#include "Base.h"
#include <Sofa/Core/Context.h>

namespace Sofa
{

// namespace Core
// {
// class Context;
// }

namespace Abstract
{
using Core::Context;

/// Base class for simulation objects.
class BaseObject : public virtual Base
{
protected:
    Context* context;
public:
    BaseObject() : context(NULL)
    {}
    virtual ~BaseObject()
    {}

    void setContext(Context* n)
    {
        context = n;
    }
    const Context* getContext() const
    {
        return context;
    }

    /// Initialization method called after each graph modification.
    virtual void init()
    { }

    /*	void setContext( Core::Context* c ){
    	    context_=c;
    	}*/
}
;

class ContextObject : public BaseObject
{
public:
    ContextObject( std::string name, Core::Context* c )
    {
        setName(name);
        setContext(c);
    }

    virtual ~ContextObject()
    {}
    /// modify the Context
    virtual void apply()=0;
protected:
    Context* getContext()
    {
        return const_cast<Context*>(this->context);
    }

};


} // namespace Abstract

} // namespace Sofa

#endif


