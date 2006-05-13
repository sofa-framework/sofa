#ifndef SOFA_ABSTRACT_BASENODE_H
#define SOFA_ABSTRACT_BASENODE_H

#include "Base.h"
#include "Sofa/Core/Context.h"

namespace Sofa
{

namespace Abstract
{

class BaseObject;

/// Base class for simulation nodes.
class BaseNode : public virtual Base, public Sofa::Core::Context
{
public:
    virtual ~BaseNode() {}

    /// @name Scene hierarchy
    /// @{

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual BaseNode* getParent() = 0;

    /// Get parent node (or NULL if no hierarchy or for root node)
    virtual const BaseNode* getParent() const = 0;

    /// Add a child node
    virtual void addChild(BaseNode* node) = 0;

    /// Remove a child node
    virtual void removeChild(BaseNode* node) = 0;

    /// Add a generic object
    virtual void addObject(BaseObject* obj) = 0;

    /// Remove a generic object
    virtual void removeObject(BaseObject* obj) = 0;

    /// @}

};

} // namespace Abstract

} // namespace Sofa

#endif
