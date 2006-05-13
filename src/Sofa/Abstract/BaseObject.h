#ifndef SOFA_ABSTRACT_BASEOBJECT_H
#define SOFA_ABSTRACT_BASEOBJECT_H

#include "BaseNode.h"

namespace Sofa
{

namespace Core
{
class Context;
}

namespace Abstract
{

/// Base class for simulation objects.
class BaseObject : public virtual Base
{
private:
    BaseNode* node;
public:
    BaseObject() : node(NULL)
    {}
    virtual ~BaseObject()
    {}

    void setNode(BaseNode* n)
    {
        node = n;
    }
    BaseNode* getNode()
    {
        return node;
    }
    const BaseNode* getNode() const
    {
        return node;
    }

    /// Initialization method called after each graph modification.
    virtual void init()
    { }

    /*	void setContext( Core::Context* c ){
    	    context_=c;
    	}*/
    Core::Context* getContext()
    {
        return node;
    }
}
;

} // namespace Abstract

} // namespace Sofa

#endif


