#ifndef SOFA_CORE_CONSTRAINT_H
#define SOFA_CORE_CONSTRAINT_H

#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class Constraint : public virtual Abstract::BaseObject
{
public:
    virtual ~Constraint() { }

    virtual void applyConstraint() = 0; ///< project dx to constrained space
};

} // namespace Core

} // namespace Sofa

#endif
