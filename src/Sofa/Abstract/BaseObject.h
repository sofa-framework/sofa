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
    const Context* context_;
public:
    BaseObject()
        : Base(), context_(NULL)
    {}
    virtual ~BaseObject()
    {}

    void setContext(const Context* n)
    {
        context_ = n;
    }
    const Context* getContext() const
    {
        return context_;
    }

    /// Initialization method called after each graph modification.
    virtual void init()
    { }

}
;


} // namespace Abstract

} // namespace Sofa

#endif
