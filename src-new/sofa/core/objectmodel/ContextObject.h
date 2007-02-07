#ifndef SOFA_CORE_OBJECTMODEL_CONTEXTOBJECT_H
#define SOFA_CORE_OBJECTMODEL_CONTEXTOBJECT_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace core
{

namespace objectmodel
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


} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif

