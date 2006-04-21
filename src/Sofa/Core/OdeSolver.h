#ifndef SOFA_CORE_ODESOLVER_H
#define SOFA_CORE_ODESOLVER_H

#include "Sofa/Abstract/BaseObject.h"
#include "IntegrationGroup.h"

namespace Sofa
{

namespace Core
{

class MechanicalGroup;

class OdeSolver : public Abstract::BaseObject
{
protected:
    IntegrationGroup* group;
public:
    OdeSolver();

    virtual ~OdeSolver();

    virtual void solve (double dt) = 0;

    virtual void setGroup(IntegrationGroup* grp);
};

} // namespace Core

} // namespace Sofa

#endif
