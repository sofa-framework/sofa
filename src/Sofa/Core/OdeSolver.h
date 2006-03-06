#ifndef SOFA_CORE_ODESOLVER_H
#define SOFA_CORE_ODESOLVER_H

#include "Sofa/Abstract/Base.h"

namespace Sofa
{

namespace Core
{

class MechanicalGroup;

class OdeSolver : public Abstract::Base
{
protected:
    MechanicalGroup* group;
public:
    OdeSolver();

    virtual ~OdeSolver();

    virtual void solve (double dt) = 0;

    virtual void setGroup(MechanicalGroup* grp);
};

} // namespace Core

} // namespace Sofa

#endif
