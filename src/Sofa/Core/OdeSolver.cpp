#include "OdeSolver.h"

#include <stdlib.h>
#include <math.h>

namespace Sofa
{

namespace Core
{

OdeSolver::OdeSolver()
    : group(NULL)
{
}

OdeSolver::~OdeSolver()
{
}

void OdeSolver::setGroup(IntegrationGroup* grp)
{
    group = grp;
}

} // namespace Core

} // namespace Sofa
