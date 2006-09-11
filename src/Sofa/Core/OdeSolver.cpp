#include "OdeSolver.h"

#include <stdlib.h>
#include <math.h>

namespace Sofa
{

namespace Core
{

OdeSolver::OdeSolver()
    : group(NULL)
    , debug_(false), mat(NULL)
{
}

OdeSolver::~OdeSolver()
{
}

void OdeSolver::setGroup(IntegrationGroup* grp)
{
    group = grp;
}

OdeSolver* OdeSolver::setDebug(bool b)
{
    debug_=b;
    return this;
}
bool OdeSolver::getDebug() const
{
    return debug_;
}

} // namespace Core

} // namespace Sofa
