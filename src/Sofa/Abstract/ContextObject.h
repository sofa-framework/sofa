#ifndef SOFA_ABSTRACT_CONTEXTOBJECT_H
#define SOFA_ABSTRACT_CONTEXTOBJECT_H

#include "BaseObject.h"

namespace Sofa
{


namespace Abstract
{

class ContextObject : public virtual BaseObject
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
    BaseContext* getContext()
    {
        return const_cast<BaseContext*>(this->context_);
    }

};


} // namespace Abstract

} // namespace Sofa

#endif

