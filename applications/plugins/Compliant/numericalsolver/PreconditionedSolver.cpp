
#include "../preconditioner/BasePreconditioner.h"
#include "../assembly/AssembledSystem.h"

#include "PreconditionedSolver.h"

#include <sofa/core/objectmodel/BaseContext.h>


namespace sofa {
namespace component {
namespace linearsolver {

PreconditionedSolver::PreconditionedSolver()
    : _preconditioner( NULL )
{}


void PreconditionedSolver::getPreconditioner( core::objectmodel::BaseContext* context )
{
    // look for an optional preconditioner
    _preconditioner = context->get<preconditioner_type>(core::objectmodel::BaseContext::Local);
}


}
}
}
