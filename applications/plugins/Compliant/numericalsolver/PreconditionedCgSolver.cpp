#include "../preconditioner/BasePreconditioner.h"
#include "../assembly/AssembledSystem.h"

#include "PreconditionedCgSolver.h"

#include <sofa/core/ObjectFactory.h>


#include "utils/kkt.h"
#include "utils/preconditionedcg.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(PreconditionedCgSolver);
int PreconditionedCgSolverClass = core::RegisterObject("Sparse PCG linear solver").add< PreconditionedCgSolver >();


void PreconditionedCgSolver::init()
{
    CgSolver::init();
    PreconditionedCgSolver::getPreconditioner( this->getContext() );
}

void PreconditionedCgSolver::solve_kkt(AssembledSystem::vec& x,
                         const AssembledSystem& system,
                         const AssembledSystem::vec& b,
                         real damping ) const {

    if( _preconditioner )
    {
        if( system.n ) {
            throw std::logic_error("CG can't solve KKT system with constraints. you need to turn on schur and add a response component for this");
        }

        params_type p = params(b);

//        report("pcg (kkt) recquired ", p );

        kkt::matrixQ A( system );
        Preconditioner P( system, _preconditioner );

        typedef ::preconditionedcg<real> solver_type;
        solver_type::solve(x, A, P, b, p);

        report("pcg (kkt)", p );
    }
    else
        CgSolver::solve_kkt( x, system, b, damping );
}


}
}
}


