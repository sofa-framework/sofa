#ifndef SOFA_ABSTRACT_BASEOBJECT_H
#define SOFA_ABSTRACT_BASEOBJECT_H

#include "Base.h"
#include "BaseContext.h"

namespace Sofa
{

namespace Abstract
{

/// Base class for simulation objects.
class BaseObject : public virtual Base
{
public:
    BaseObject()
        : Base(), context_(NULL)
    {}

    virtual ~BaseObject()
    {}

    void setContext(BaseContext* n)
    {
        context_ = n;
    }

    const BaseContext* getContext() const
    {
        return (context_==NULL)?BaseContext::getDefault():context_;
    }

    BaseContext* getContext()
    {
        return (context_==NULL)?BaseContext::getDefault():context_;
    }

    /// Initialization method called after each graph modification.
    virtual void init()
    { }

    /// Reset to initial state
    virtual void reset()
    { }

protected:
    BaseContext* context_;
};

} // namespace Abstract

} // namespace Sofa

#endif
