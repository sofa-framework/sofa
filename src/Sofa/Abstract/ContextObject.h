#ifndef SOFA_ABSTRACT_CONTEXTOBJECT_H
#define SOFA_ABSTRACT_CONTEXTOBJECT_H

#include "BaseObject.h"

namespace Sofa
{


namespace Abstract
{
using Core::Context;


class ContextObject : public BaseObject
{
public:
    ContextObject()
        : BaseObject()
    {}

    virtual ~ContextObject()
    {}

    /// modify the Context
    virtual void apply()=0;
protected:
    Context* getContext()
    {
        return const_cast<Context*>(this->context_);
    }

};


} // namespace Abstract

} // namespace Sofa

#endif

