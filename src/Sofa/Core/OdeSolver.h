#ifndef SOFA_CORE_ODESOLVER_H
#define SOFA_CORE_ODESOLVER_H

#include "Sofa/Abstract/BaseObject.h"
#include "Sofa/Components/Common/SofaBaseMatrix.h"
//#include "IntegrationGroup.h"

namespace Sofa
{

namespace Core
{

class MechanicalGroup;

class OdeSolver : public Abstract::BaseObject
{
public:
    OdeSolver();

    virtual ~OdeSolver();

    virtual void solve (double dt) = 0;

    //virtual void setGroup(Abstract::BaseContext* grp);

protected:
    //IntegrationGroup* group;
    //Abstract::BaseContext* group;
    Components::Common::SofaBaseMatrix *mat;


};

} // namespace Core

} // namespace Sofa

#endif

