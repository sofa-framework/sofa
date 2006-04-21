#ifndef SOFA_ABSTRACT_BASENODE_H
#define SOFA_ABSTRACT_BASENODE_H

#include "Base.h"

namespace Sofa
{

namespace Abstract
{

class BaseObject;

/// Base class for simulation nodes.
class BaseNode : public virtual Base
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

    /// @name Global Parameters
    /// @{

    /// Gravity vector as a pointer to 3 double
    virtual const double* getGravity() const = 0;

    /// Animation flag
    virtual bool getAnimate() const = 0;

    /// MultiThreading activated
    virtual bool getMultiThreadSimulation() const = 0;

    /// Display flags: Collision Models
    virtual bool getShowCollisionModels() const = 0;

    /// Display flags: Behavior Models
    virtual bool getShowBehaviorModels() const = 0;

    /// Display flags: Visual Models
    virtual bool getShowVisualModels() const = 0;

    /// Display flags: Mappings
    virtual bool getShowMappings() const = 0;

    /// Display flags: ForceFields
    virtual bool getShowForceFields() const = 0;

    /// @}
};

} // namespace Abstract

} // namespace Sofa

#endif
